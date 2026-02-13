"""
PersonaPlex 实时翻译 - 云端 GPU 版本
自动使用 CUDA，优化云端部署
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

warnings.filterwarnings("ignore")

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

# 全局变量
model_loaded = False

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

def load_personaplex_model():
    """加载 PersonaPlex 模型"""
    global model_loaded
    try:
        import moshi.offline as moshi_offline
        print(f"正在初始化 PersonaPlex")
        print(f"使用设备: {device}")
        
        # 清理内存
        clear_memory()
        
        model_loaded = True
        print("✓ PersonaPlex 已就绪")
        return True
    except ImportError as e:
        print(f"✗ 无法导入 moshi 包: {e}")
        print("   请安装: pip install -e personaplex/moshi/")
        return False

def process_audio_chunk(audio_data, text_prompt, source_lang, target_lang):
    """处理音频块 - 云端 GPU 版本"""
    global is_processing
    
    # 检查音频长度（限制最大长度）
    max_samples = 24000 * 15  # 最多15秒
    if len(audio_data) > max_samples:
        print(f"⚠️  音频太长 ({len(audio_data)} 采样点)，截断到 {max_samples}")
        audio_data = audio_data[:max_samples]
    
    try:
        import moshi.offline as moshi_offline
        
        # 清理内存
        clear_memory()
        
        # 创建临时文件
        input_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        text_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        
        input_path = input_temp.name
        output_path = output_temp.name
        text_path = text_temp.name
        
        # 保存输入音频
        sf.write(input_path, audio_data, 24000)
        input_temp.close()
        output_temp.close()
        text_temp.close()
        
        # 云端 GPU 使用 CUDA（不需要 CPU offload）
        if torch.cuda.is_available():
            moshi_device = "cuda"
            use_cpu_offload = False
            print(f"✓ 使用 CUDA GPU 模式")
        else:
            # 如果没有 GPU，使用 CPU + CPU offload
            moshi_device = "cpu"
            try:
                import accelerate
                use_cpu_offload = True
                print("⚠️  使用 CPU + CPU offload 模式（较慢）")
            except ImportError:
                use_cpu_offload = False
                print("⚠️  使用 CPU 模式（未安装 accelerate，可能较慢）")
        
        # 获取 voice prompt 目录
        try:
            voice_prompt_dir = moshi_offline._get_voice_prompt_dir(None, "nvidia/personaplex-7b-v1")
            voice_prompt = os.path.join(voice_prompt_dir, "NATF2.pt")
            if not os.path.exists(voice_prompt):
                voice_prompt = "NATF2.pt"
        except:
            voice_prompt = "NATF2.pt"
        
        # 确保使用 HF_TOKEN
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
            except Exception as e:
                print(f"⚠️  登录 Hugging Face 失败: {e}")
        else:
            print("⚠️  警告: 未设置 HF_TOKEN")
        
        print(f"开始处理音频（{len(audio_data)} 采样点，约 {len(audio_data)/24000:.1f} 秒）...")
        start_time = time.time()
        
        # 调用 moshi offline 推理
        moshi_offline.run_inference(
            input_wav=input_path,
            output_wav=output_path,
            output_text=text_path,
            text_prompt=text_prompt,
            voice_prompt_path=voice_prompt,
            tokenizer_path=None,
            moshi_weight=None,
            mimi_weight=None,
            hf_repo="nvidia/personaplex-7b-v1",
            device=moshi_device,
            seed=42424242,
            temp_audio=0.8,
            temp_text=0.7,
            topk_audio=250,
            topk_text=25,
            greedy=False,
            save_voice_prompt_embeddings=False,
            cpu_offload=use_cpu_offload,
        )
        
        elapsed = time.time() - start_time
        print(f"✓ 处理完成（耗时 {elapsed:.1f} 秒）")
        
        # 读取输出音频
        audio_output, sr = librosa.load(output_path, sr=24000)
        
        # 清理临时文件
        os.unlink(input_path)
        os.unlink(output_path)
        os.unlink(text_path)
        
        # 清理内存
        clear_memory()
        
        return audio_output
    except Exception as e:
        print(f"处理错误: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()
        return None
    finally:
        is_processing = False

def process_queue():
    """处理队列中的请求"""
    while True:
        try:
            item = processing_queue.get(timeout=30)
            if item is None:
                break
            
            audio_data, text_prompt, source_lang, target_lang, callback = item
            
            translated_audio = process_audio_chunk(audio_data, text_prompt, source_lang, target_lang)
            
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
        'model_loaded': model_loaded,
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'cuda_info': cuda_info,
        'queue_size': processing_queue.qsize(),
        'is_processing': is_processing
    })

@app.route('/api/load_model', methods=['POST'])
def load_model():
    global model_loaded
    if model_loaded:
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
            audio_data, sr = librosa.load(temp_path, sr=24000)
            print(f"收到音频: {len(audio_data)} 采样点 ({len(audio_data)/24000:.1f} 秒)")
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
        
        # 创建翻译提示词（使用 PersonaPlex 的 system 标签格式）
        lang_names = {
            "en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French",
            "de": "German", "ja": "Japanese", "ko": "Korean"
        }
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        # 使用 <system> 标签包裹，让模型明确这是翻译任务
        text_prompt = f"""<system> You are a professional real-time translator. Your ONLY job is to translate speech from {source_name} to {target_name}. Do NOT introduce yourself. Do NOT have conversations. ONLY translate what the user says. Always respond in {target_name} only. Never speak in {source_name}. <system>"""
        
        # 定义回调函数
        def send_result(translated_audio):
            if translated_audio is not None:
                try:
                    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    sf.write(output_temp.name, translated_audio, 24000)
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
                socketio.emit('audio_error', {'error': 'Translation failed'})
        
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
    print("PersonaPlex 实时翻译 - 云端 GPU 版本")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"✓ 使用 CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  未检测到 CUDA GPU，将使用 CPU（较慢）")
    print("=" * 60)
    print(f"启动服务器在端口 {port}")
    print("")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

