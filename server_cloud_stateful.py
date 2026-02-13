"""
PersonaPlex 实时翻译 - 云端 GPU 版本（保持模型状态）
在启动时加载模型一次，保持状态，避免每次重新加载
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import torch
import numpy as np
import soundfile as sf
import librosa
import os
import warnings
import tempfile
import base64
import threading
import gc
import queue
import time
import sentencepiece

warnings.filterwarnings("ignore")

# 设置 PyTorch CUDA 内存分配配置，减少内存碎片
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# 检查并设置 Hugging Face Token
if not os.environ.get('HF_TOKEN'):
    print("⚠️  警告: HF_TOKEN 环境变量未设置")
    print("   请设置 Hugging Face Token:")
    print("   export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>")
    print("")
else:
    os.environ['HUGGING_FACE_HUB_TOKEN'] = os.environ['HF_TOKEN']
    print(f"✓ HF_TOKEN 已设置 (长度: {len(os.environ['HF_TOKEN'])} 字符)")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局变量 - 模型状态
model_state = None
model_lock = threading.Lock()

# 自动检测设备（优先 CUDA，云端 GPU 使用）
device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"✓ 检测到 CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print(f"⚠️  未检测到 CUDA GPU，使用设备: {device}")

# 处理队列（限制并发，避免内存溢出）
processing_queue = queue.Queue(maxsize=2)  # 最多2个请求排队
is_processing = False

def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def wrap_with_system_tags(text: str) -> str:
    """包装系统提示词"""
    cleaned = text.strip()
    if not cleaned:
        return ""
    return f"<system> {cleaned} <system>"

def warmup(mimi, other_mimi, lm_gen, device, frame_size):
    """预热模型"""
    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c: c + 1])
            if tokens is None:
                continue
            _ = mimi.decode(tokens[:, 1:9])
            _ = other_mimi.decode(tokens[:, 1:9])
    
    if device == "cuda" or (isinstance(device, torch.device) and device.type == 'cuda'):
        torch.cuda.synchronize()

def load_personaplex_model():
    """加载 PersonaPlex 模型并保持状态"""
    global model_state
    
    try:
        from moshi.models.loaders import get_mimi, get_moshi_lm, MIMI_NAME, TEXT_TOKENIZER_NAME, MOSHI_NAME
        from moshi.models.lm import LMGen
        from moshi.offline import _get_voice_prompt_dir
        from huggingface_hub import hf_hub_download
        
        print(f"正在加载 PersonaPlex 模型...")
        print(f"使用设备: {device}")
        
        # 清理内存
        clear_memory()
        
        hf_repo = "nvidia/personaplex-7b-v1"
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        
        # 确保使用 HF_TOKEN
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
            except Exception as e:
                print(f"⚠️  登录 Hugging Face 失败: {e}")
        
        # 下载 config.json 以增加下载计数
        hf_hub_download(hf_repo, "config.json", token=hf_token)
        
        # 1) 加载 Mimi 编码器/解码器
        print("正在加载 Mimi...")
        mimi_weight = hf_hub_download(hf_repo, MIMI_NAME, token=hf_token)
        mimi = get_mimi(mimi_weight, device)
        other_mimi = get_mimi(mimi_weight, device)
        print("✓ Mimi 已加载")
        
        # 2) 加载 tokenizer
        print("正在加载 tokenizer...")
        tokenizer_path = hf_hub_download(hf_repo, TEXT_TOKENIZER_NAME, token=hf_token)
        text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        print("✓ Tokenizer 已加载")
        
        # 3) 加载 Moshi LM
        print("正在加载 Moshi LM...")
        moshi_weight = hf_hub_download(hf_repo, MOSHI_NAME, token=hf_token)
        use_cpu_offload = False if torch.cuda.is_available() else True
        lm = get_moshi_lm(moshi_weight, device=device, cpu_offload=use_cpu_offload)
        lm.eval()
        print("✓ Moshi LM 已加载")
        
        # 4) 创建 LMGen
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
            sample_rate=mimi.sample_rate,
            device=device,
            frame_rate=mimi.frame_rate,
            save_voice_prompt_embeddings=False,
            use_sampling=True,
            temp=0.8,
            temp_text=0.7,
            top_k=250,
            top_k_text=25,
        )
        
        # 保持流式状态
        mimi.streaming_forever(1)
        other_mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)
        
        # 5) 预热
        print("正在预热模型...")
        warmup(mimi, other_mimi, lm_gen, device, frame_size)
        print("✓ 模型预热完成")
        
        # 获取 voice prompt 目录
        voice_prompt_dir = _get_voice_prompt_dir(None, hf_repo)
        
        # 保存模型状态
        model_state = {
            'mimi': mimi,
            'other_mimi': other_mimi,
            'text_tokenizer': text_tokenizer,
            'lm_gen': lm_gen,
            'device': device,
            'frame_size': frame_size,
            'voice_prompt_dir': voice_prompt_dir,
            'sample_rate': mimi.sample_rate,
        }
        
        print("✓ PersonaPlex 模型已加载并保持状态")
        return True
        
    except ImportError as e:
        print(f"✗ 无法导入 moshi 包: {e}")
        print("   请安装: pip install -e personaplex/moshi/")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_audio_chunk(audio_data, text_prompt, voice_prompt_path=None):
    """处理音频块 - 使用已加载的模型状态"""
    global model_state, is_processing
    
    if model_state is None:
        print("✗ 模型未加载")
        return None
    
    # 检查音频长度（限制最大长度，减少内存使用）
    max_samples = model_state['sample_rate'] * 5  # 最多5秒（减少内存压力）
    if len(audio_data) > max_samples:
        print(f"⚠️  音频太长 ({len(audio_data)} 采样点)，截断到 {max_samples}")
        audio_data = audio_data[:max_samples]
    
    # 处理前清理 CUDA 缓存
    clear_memory()
    
    try:
        with model_lock:
            mimi = model_state['mimi']
            other_mimi = model_state['other_mimi']
            text_tokenizer = model_state['text_tokenizer']
            lm_gen = model_state['lm_gen']
            device = model_state['device']
            frame_size = model_state['frame_size']
            sample_rate = model_state['sample_rate']
            
            # 重置流式状态（开始新对话）
            mimi.reset_streaming()
            other_mimi.reset_streaming()
            lm_gen.reset_streaming()
            
            # 设置 text prompt
            if text_prompt:
                wrapped_prompt = wrap_with_system_tags(text_prompt)
                lm_gen.text_prompt_tokens = text_tokenizer.encode(wrapped_prompt) if wrapped_prompt else None
            else:
                lm_gen.text_prompt_tokens = None
            
            # 设置 voice prompt
            if voice_prompt_path is None:
                voice_prompt_dir = model_state['voice_prompt_dir']
                voice_prompt_path = os.path.join(voice_prompt_dir, "NATF2.pt")
                if not os.path.exists(voice_prompt_path):
                    # 尝试其他路径
                    voice_prompt_path = "NATF2.pt"
            
            if os.path.exists(voice_prompt_path):
                if voice_prompt_path.endswith('.pt'):
                    lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
                else:
                    lm_gen.load_voice_prompt(voice_prompt_path)
            
            # 运行系统提示阶段
            lm_gen.step_system_prompts(mimi)
            mimi.reset_streaming()  # 重置 mimi 流式状态
            
            print(f"开始处理音频（{len(audio_data)} 采样点，约 {len(audio_data)/sample_rate:.1f} 秒）...")
            start_time = time.time()
            
            # 处理音频帧
            generated_frames = []
            # 确保音频数据是 float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            audio_tensor = torch.from_numpy(audio_data).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # (1, T)
            audio_tensor = audio_tensor.to(device)
            
            # 将音频分成帧并处理（保持 float32）
            all_pcm_data = audio_tensor[0].cpu().numpy().astype(np.float32)
            del audio_tensor  # 释放内存
            
            frame_count = 0
            while all_pcm_data.shape[-1] >= frame_size:
                chunk = all_pcm_data[:frame_size]
                all_pcm_data = all_pcm_data[frame_size:]
                
                # 明确指定 dtype 为 float32
                chunk_tensor = torch.from_numpy(chunk.astype(np.float32)).float().to(device)[None, None]  # (1, 1, frame_size)
                
                # 编码
                codes = mimi.encode(chunk_tensor)
                _ = other_mimi.encode(chunk_tensor)
                del chunk_tensor  # 释放内存
                
                # 逐步处理每个 codebook
                for c in range(codes.shape[-1]):
                    tokens = lm_gen.step(codes[:, :, c: c + 1])
                    if tokens is None:
                        continue
                    
                    # 解码音频
                    pcm = mimi.decode(tokens[:, 1:9])
                    _ = other_mimi.decode(tokens[:, 1:9])
                    pcm = pcm.detach().cpu().numpy()[0, 0]
                    generated_frames.append(pcm)
                    del pcm  # 释放 GPU 内存
                
                del codes  # 释放内存
                frame_count += 1
                
                # 每处理10帧清理一次缓存
                if frame_count % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # 处理剩余的音频
            if all_pcm_data.shape[-1] > 0:
                # 填充到 frame_size（确保 float32）
                padding = np.zeros(frame_size - all_pcm_data.shape[-1], dtype=np.float32)
                chunk = np.concatenate([all_pcm_data, padding])
                chunk_tensor = torch.from_numpy(chunk.astype(np.float32)).float().to(device)[None, None]
                codes = mimi.encode(chunk_tensor)
                _ = other_mimi.encode(chunk_tensor)
                del chunk_tensor
                for c in range(codes.shape[-1]):
                    tokens = lm_gen.step(codes[:, :, c: c + 1])
                    if tokens is None:
                        continue
                    pcm = mimi.decode(tokens[:, 1:9])
                    _ = other_mimi.decode(tokens[:, 1:9])
                    pcm = pcm.detach().cpu().numpy()[0, 0]
                    generated_frames.append(pcm)
                    del pcm
                del codes
            
            # 合并所有生成的帧
            if generated_frames:
                output_audio = np.concatenate(generated_frames)
            else:
                output_audio = np.array([], dtype=np.float32)
            
            # 清理内存
            del generated_frames
            del all_pcm_data
            clear_memory()
            
            elapsed = time.time() - start_time
            print(f"✓ 处理完成（耗时 {elapsed:.1f} 秒）")
            
            return output_audio
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"✗ GPU 内存不足: {e}")
        print("正在清理内存...")
        clear_memory()
        # 尝试再次清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        return None
    except Exception as e:
        print(f"处理错误: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()
        return None
    finally:
        is_processing = False
        # 最后清理一次
        clear_memory()

def process_queue():
    """处理队列中的请求"""
    while True:
        try:
            item = processing_queue.get(timeout=30)
            if item is None:
                break
            
            audio_data, text_prompt, source_lang, target_lang, callback = item
            
            translated_audio = process_audio_chunk(audio_data, text_prompt)
            
            if callback:
                callback(translated_audio)
            
            processing_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"队列处理错误: {e}")
            import traceback
            traceback.print_exc()

# 启动队列处理线程
queue_thread = threading.Thread(target=process_queue, daemon=True)
queue_thread.start()

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    cuda_info = {}
    if torch.cuda.is_available():
        cuda_info = {
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        }
    
    return jsonify({
        'model_loaded': model_state is not None,
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'cuda_info': cuda_info,
        'queue_size': processing_queue.qsize(),
        'is_processing': is_processing
    })

@app.route('/api/load_model', methods=['POST'])
def load_model():
    if model_state is not None:
        return jsonify({'success': True, 'message': '模型已加载'})
    
    success = load_personaplex_model()
    if success:
        return jsonify({'success': True, 'message': '模型加载成功'})
    else:
        return jsonify({'success': False, 'message': '模型加载失败'}), 500

@socketio.on('connect')
def handle_connect():
    print('客户端已连接')

@socketio.on('disconnect')
def handle_disconnect():
    print('客户端已断开')

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """处理实时音频块"""
    try:
        audio_array = data.get('audio')
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'zh')
        
        if audio_array is None or not isinstance(audio_array, list) or len(audio_array) == 0:
            print("⚠️  无效的音频数据")
            socketio.emit('audio_error', {'error': 'Invalid audio data'})
            return
        
        if model_state is None:
            print("⚠️  模型未加载")
            socketio.emit('audio_error', {'error': 'Model not loaded'})
            return
        
        # 转换为 bytes
        audio_bytes = bytes(audio_array)
        
        # 验证文件头
        if len(audio_bytes) < 4 or audio_bytes[:4] != b'RIFF':
            print("⚠️  不是有效的 WAV 文件")
            socketio.emit('audio_error', {'error': 'Invalid WAV file'})
            return
        
        # 保存为临时文件并加载
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_input.write(audio_bytes)
        temp_input.close()
        temp_path = temp_input.name
        
        try:
            # 明确指定 dtype 为 float32
            audio_data, sr = librosa.load(temp_path, sr=model_state['sample_rate'], dtype=np.float32)
            print(f"收到音频: {len(audio_data)} 采样点 ({len(audio_data)/sr:.1f} 秒)")
        except Exception as e:
            print(f"音频加载错误: {e}")
            socketio.emit('audio_error', {'error': f'Audio load error: {str(e)}'})
            return
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # 检查队列是否已满
        if processing_queue.full():
            print("⚠️  处理队列已满，跳过此请求")
            socketio.emit('audio_error', {'error': 'Processing queue is full, please wait'})
            return
        
        # 创建提示词 - 简单对话测试
        text_prompt = "You enjoy having a good conversation."
        
        # 定义回调函数
        def send_result(translated_audio):
            if translated_audio is not None and len(translated_audio) > 0:
                try:
                    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    sf.write(output_temp.name, translated_audio, model_state['sample_rate'])
                    output_temp.close()
                    
                    with open(output_temp.name, 'rb') as f:
                        audio_bytes = f.read()
                    os.unlink(output_temp.name)
                    
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    socketio.emit('translated_audio', {'audio': audio_base64})
                    print("✓ 已发送翻译结果")
                except Exception as e:
                    print(f"发送结果错误: {e}")
                    socketio.emit('audio_error', {'error': f'Failed to send result: {str(e)}'})
            else:
                socketio.emit('audio_error', {'error': 'Translation failed or empty result'})
        
        # 添加到处理队列
        try:
            processing_queue.put_nowait((audio_data, text_prompt, source_lang, target_lang, send_result))
            print(f"✓ 已添加到处理队列（队列大小: {processing_queue.qsize()}）")
        except queue.Full:
            print("⚠️  队列已满")
            socketio.emit('audio_error', {'error': 'Processing queue is full'})
        
    except Exception as e:
        print(f"处理音频块错误: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('audio_error', {'error': f'Server error: {str(e)}'})

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    
    print("=" * 60)
    print("PersonaPlex 实时翻译 - 云端 GPU 版本（保持模型状态）")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"✓ 使用 CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  未检测到 CUDA GPU，将使用 CPU（较慢）")
    print("=" * 60)
    
    # 启动时自动加载模型
    print("正在加载模型...")
    load_personaplex_model()
    
    print(f"启动服务器在端口 {port}")
    print("")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

