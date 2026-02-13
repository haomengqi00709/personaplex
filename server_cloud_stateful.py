"""
PersonaPlex 实时对话 - 云端 GPU 版本（保持模型状态）
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
# 优化 Socket.IO 配置，减少连接问题
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    ping_timeout=60,  # 增加 ping 超时时间
    ping_interval=25,  # 增加 ping 间隔
    max_http_buffer_size=10*1024*1024,  # 10MB 缓冲区
    logger=False,  # 关闭 Socket.IO 内部日志（减少噪音）
    engineio_logger=False
)

# 全局变量 - 模型状态
model_state = None
model_lock = threading.Lock()
conversation_active = False  # 跟踪是否有活跃对话
last_audio_time = 0  # 上次处理音频的时间

# 调试统计
debug_stats = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'total_processing_time': 0.0,
    'last_request_time': None,
    'last_processing_time': 0.0,
    'memory_usage_mb': 0.0,
}

# 自动检测设备（优先 CUDA，云端 GPU 使用）
device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"✓ 检测到 CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print(f"⚠️  未检测到 CUDA GPU，使用设备: {device}")

# 处理队列（限制并发，避免内存溢出）
processing_queue = queue.Queue(maxsize=1)  # 最多1个请求排队
is_processing = False
last_request_id = 0  # 请求ID，用于去重
pending_request_time = 0  # 待处理请求的时间戳

def get_memory_usage():
    """获取内存使用情况（MB）"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return {
            'allocated_mb': round(allocated, 2),
            'reserved_mb': round(reserved, 2),
            'free_mb': round((torch.cuda.get_device_properties(0).total_memory / 1024**2) - reserved, 2)
        }
    return {'allocated_mb': 0, 'reserved_mb': 0, 'free_mb': 0}

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
    global model_state, debug_stats, last_audio_time
    
    request_start_time = time.time()
    debug_stats['total_requests'] += 1
    debug_stats['last_request_time'] = time.strftime('%H:%M:%S')
    
    if model_state is None:
        print("✗ [ERROR] 模型未加载")
        debug_stats['failed_requests'] += 1
        return None
    
    # 记录内存使用
    mem_info = get_memory_usage()
    debug_stats['memory_usage_mb'] = mem_info['allocated_mb']
    print(f"📊 [DEBUG] 请求 #{debug_stats['total_requests']} | 内存: {mem_info['allocated_mb']:.1f}MB / {mem_info['reserved_mb']:.1f}MB | 可用: {mem_info['free_mb']:.1f}MB")
    
    # 移除音频长度限制，允许完整处理
    # 只保留极端情况的安全检查（超过60秒可能是错误）
    max_samples_safety = model_state['sample_rate'] * 60  # 安全上限：60秒（防止极端情况）
    if len(audio_data) > max_samples_safety:
        print(f"⚠️  [WARN] 音频异常长 ({len(audio_data)} 采样点，{len(audio_data)/model_state['sample_rate']:.2f}秒)，可能是错误，截断到安全上限 {max_samples_safety} ({max_samples_safety/model_state['sample_rate']:.2f}秒)")
        audio_data = audio_data[:max_samples_safety]
    else:
        print(f"✓ [AUDIO] 音频长度: {len(audio_data)} 采样点 ({len(audio_data)/model_state['sample_rate']:.2f}秒) - 完整处理")
    
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
            
            global conversation_active, last_audio_time
            current_time = time.time()
            
            # 如果距离上次处理超过30秒，认为是新对话（大幅增加时间窗口，减少重新初始化）
            time_since_last = current_time - last_audio_time if last_audio_time > 0 else 999
            is_new_conversation = not conversation_active or time_since_last > 30.0
            
            if not is_new_conversation:
                print(f"⏱️  [TIME] 距离上次请求: {time_since_last:.1f}秒（继续对话，跳过初始化）")
            
            if is_new_conversation:
                print(f"🔄 [NEW_CONV] 开始新对话 #{debug_stats['total_requests']}，初始化系统提示...")
                init_start = time.time()
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
                
                # 运行系统提示阶段（只在新对话时运行，这是最耗时的步骤）
                lm_gen.step_system_prompts(mimi)
                mimi.reset_streaming()  # 重置 mimi 流式状态
                conversation_active = True
                init_time = time.time() - init_start
                print(f"✓ [INIT] 系统提示初始化完成，耗时: {init_time:.2f}秒")
            else:
                print(f"➡️  [CONTINUE] 继续对话 #{debug_stats['total_requests']}，跳过系统提示初始化（节省约2.3秒）")
                # 继续对话，只重置流式状态，不重新运行系统提示
                mimi.reset_streaming()
                other_mimi.reset_streaming()
                lm_gen.reset_streaming()
            
            audio_duration = len(audio_data) / sample_rate
            print(f"🎤 [AUDIO] 开始处理音频 | 采样点: {len(audio_data)} | 时长: {audio_duration:.2f}秒")
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
            encode_time = 0
            decode_time = 0
            step_time = 0
            
            while all_pcm_data.shape[-1] >= frame_size:
                chunk = all_pcm_data[:frame_size]
                all_pcm_data = all_pcm_data[frame_size:]
                
                # 明确指定 dtype 为 float32
                chunk_tensor = torch.from_numpy(chunk.astype(np.float32)).float().to(device)[None, None]  # (1, 1, frame_size)
                
                # 编码
                encode_start = time.time()
                codes = mimi.encode(chunk_tensor)
                _ = other_mimi.encode(chunk_tensor)
                encode_time += time.time() - encode_start
                del chunk_tensor  # 释放内存
                
                # 逐步处理每个 codebook
                for c in range(codes.shape[-1]):
                    step_start = time.time()
                    tokens = lm_gen.step(codes[:, :, c: c + 1])
                    step_time += time.time() - step_start
                    if tokens is None:
                        continue
                    
                    # 解码音频
                    decode_start = time.time()
                    pcm = mimi.decode(tokens[:, 1:9])
                    _ = other_mimi.decode(tokens[:, 1:9])
                    decode_time += time.time() - decode_start
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
            
            if frame_count > 0:
                print(f"📈 [PROCESS] 处理了 {frame_count} 帧 | 编码: {encode_time:.2f}s | 推理: {step_time:.2f}s | 解码: {decode_time:.2f}s")
            
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
            total_time = time.time() - request_start_time
            output_duration = len(output_audio) / sample_rate if len(output_audio) > 0 else 0
            
            debug_stats['last_processing_time'] = elapsed
            debug_stats['total_processing_time'] += elapsed
            debug_stats['successful_requests'] += 1
            
            # 更新内存信息
            mem_info = get_memory_usage()
            print(f"✓ [DONE] 处理完成 | 总耗时: {total_time:.2f}s | 处理耗时: {elapsed:.2f}s | 输出时长: {output_duration:.2f}s")
            print(f"📊 [MEMORY] 处理后内存: {mem_info['allocated_mb']:.1f}MB / {mem_info['reserved_mb']:.1f}MB | 可用: {mem_info['free_mb']:.1f}MB")
            
            # 在处理完成后更新 last_audio_time（这样下次请求时，时间窗口更准确）
            last_audio_time = time.time()
            
            return output_audio
            
    except torch.cuda.OutOfMemoryError as e:
        debug_stats['failed_requests'] += 1
        mem_info = get_memory_usage()
        print(f"✗ [OOM] GPU 内存不足 | 已分配: {mem_info['allocated_mb']:.1f}MB | 已保留: {mem_info['reserved_mb']:.1f}MB")
        print(f"   [OOM] 错误详情: {str(e)[:200]}")
        print("   [OOM] 正在清理内存...")
        clear_memory()
        # 尝试再次清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        return None
    except Exception as e:
        debug_stats['failed_requests'] += 1
        print(f"✗ [ERROR] 处理错误: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        clear_memory()
        return None
    finally:
        # 注意：is_processing 在 process_queue 中管理，这里不重置
        # 最后清理一次
        clear_memory()

def process_queue():
    """处理队列中的请求"""
    global is_processing
    while True:
        try:
            item = processing_queue.get(timeout=30)
            if item is None:
                break
            
            # 标记开始处理
            is_processing = True
            process_start = time.time()
            
            audio_data, text_prompt, source_lang, target_lang, callback = item
            
            # 处理音频
            print(f"🔄 [QUEUE] 开始处理队列中的请求...")
            response_audio = process_audio_chunk(audio_data, text_prompt)
            
            process_time = time.time() - process_start
            
            # 回调发送结果
            if callback:
                if response_audio is not None and len(response_audio) > 0:
                    print(f"📤 [SEND] 发送响应音频，长度: {len(response_audio)} 采样点 | 总处理时间: {process_time:.2f}秒")
                else:
                    print(f"⚠️  [SEND] 响应音频为空，不发送")
                callback(response_audio)
            
            # 标记处理完成
            is_processing = False
            processing_queue.task_done()
            
            # 清空队列中等待的其他请求（避免堆积，只处理最新的）
            while not processing_queue.empty():
                try:
                    old_item = processing_queue.get_nowait()
                    print(f"🗑️  [CLEAR] 丢弃队列中的旧请求（避免堆积）")
                    processing_queue.task_done()
                except queue.Empty:
                    break
            
            # 短暂延迟
            time.sleep(0.1)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"队列处理错误: {e}")
            import traceback
            traceback.print_exc()
            is_processing = False

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
    
    mem_info = get_memory_usage()
    avg_processing_time = 0.0
    if debug_stats['successful_requests'] > 0:
        avg_processing_time = debug_stats['total_processing_time'] / debug_stats['successful_requests']
    
    return jsonify({
        'model_loaded': model_state is not None,
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'cuda_info': cuda_info,
        'queue_size': processing_queue.qsize(),
        'is_processing': is_processing,
        'conversation_active': conversation_active,
        'debug_stats': {
            **debug_stats,
            'avg_processing_time': round(avg_processing_time, 2),
            'memory_info': mem_info
        }
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
    print(f'🔌 [CONNECT] 客户端已连接 | 时间: {time.strftime("%H:%M:%S")}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'🔌 [DISCONNECT] 客户端已断开 | 时间: {time.strftime("%H:%M:%S")}')

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
            audio_duration = len(audio_data) / sr
            print(f"📥 [RECEIVE] 收到音频 | 采样点: {len(audio_data)} | 时长: {audio_duration:.2f}秒 | 时间: {time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"音频加载错误: {e}")
            socketio.emit('audio_error', {'error': f'Audio load error: {str(e)}'})
            return
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # 检查是否正在处理 - 如果正在处理，直接丢弃新请求（不排队，避免堆积）
        global is_processing, pending_request_time
        current_time = time.time()
        
        if is_processing:
            # 如果正在处理，且距离上次请求不到5秒，直接丢弃（避免堆积）
            if current_time - pending_request_time < 5.0:
                print(f"⚠️  [SKIP] 正在处理中，丢弃此请求（避免堆积）| 距离上次请求: {current_time - pending_request_time:.1f}秒")
                socketio.emit('audio_error', {'error': 'Processing, please wait'})
                return
            else:
                # 如果处理时间太长（超过5秒），可能是卡住了，允许新请求
                print(f"⚠️  [WARN] 处理时间过长，允许新请求")
        
        # 检查队列是否已满
        if processing_queue.full():
            print(f"⚠️  [SKIP] 队列已满，跳过此请求")
            socketio.emit('audio_error', {'error': 'Queue is full, please wait'})
            return
        
        pending_request_time = current_time
        
        # 创建提示词 - 更明确的对话指令
        text_prompt = "You are a helpful and friendly conversational AI. Respond naturally to what the user says. Do not introduce yourself or say hello unless the user greets you first. Keep your responses concise and relevant to the conversation."
        
        # 定义回调函数
        def send_result(response_audio):
            send_start = time.time()
            if response_audio is not None and len(response_audio) > 0:
                try:
                    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    sf.write(output_temp.name, response_audio, model_state['sample_rate'])
                    output_temp.close()
                    
                    with open(output_temp.name, 'rb') as f:
                        audio_bytes = f.read()
                    os.unlink(output_temp.name)
                    
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    response_duration = len(response_audio) / model_state['sample_rate']
                    socketio.emit('translated_audio', {'audio': audio_base64})
                    send_time = time.time() - send_start
                    print(f"✓ [SENT] 已发送对话响应 | 时长: {response_duration:.2f}秒 | 大小: {len(audio_bytes)} 字节 | 发送耗时: {send_time:.3f}秒")
                except Exception as e:
                    print(f"✗ [SEND_ERROR] 发送结果错误: {type(e).__name__}: {str(e)}")
                    socketio.emit('audio_error', {'error': f'Failed to send result: {str(e)}'})
            else:
                print(f"⚠️  [SEND_ERROR] 响应音频为空，不发送")
                socketio.emit('audio_error', {'error': 'Response failed or empty result'})
        
        # 添加到处理队列
        try:
            processing_queue.put_nowait((audio_data, text_prompt, source_lang, target_lang, send_result))
            print(f"✓ [QUEUE] 已添加到处理队列 | 队列大小: {processing_queue.qsize()} | 等待处理...")
        except queue.Full:
            print(f"⚠️  [QUEUE] 队列已满，无法添加新请求")
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
    print("PersonaPlex 实时对话 - 云端 GPU 版本（保持模型状态）")
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

