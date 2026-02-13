"""
PersonaPlex 实时翻译 - 简化版本
直接使用二进制数组传输，避免 base64 编码问题
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

warnings.filterwarnings("ignore")

# 检查并设置 Hugging Face Token
if not os.environ.get('HF_TOKEN'):
    print("⚠️  警告: HF_TOKEN 环境变量未设置")
    print("   请设置 Hugging Face Token:")
    print("   export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>")
    print("")
    print("   获取 Token:")
    print("   1. 访问 https://huggingface.co/settings/tokens")
    print("   2. 创建新的 token (需要 read 权限)")
    print("   3. ⚠️  重要：勾选 'Enable access to public gated repositories'")
    print("   4. 接受 PersonaPlex 模型许可: https://huggingface.co/nvidia/personaplex-7b-v1")
    print("")
else:
    # 确保 huggingface_hub 使用这个 token
    os.environ['HUGGING_FACE_HUB_TOKEN'] = os.environ['HF_TOKEN']
    print(f"✓ HF_TOKEN 已设置 (长度: {len(os.environ['HF_TOKEN'])} 字符)")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局变量
model_loaded = False
device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")

def load_personaplex_model():
    """加载 PersonaPlex 模型"""
    global model_loaded
    try:
        import moshi.offline as moshi_offline
        print(f"正在初始化 PersonaPlex")
        print(f"使用设备: {device}")
        model_loaded = True
        print("✓ PersonaPlex 已就绪")
        return True
    except ImportError as e:
        print(f"✗ 无法导入 moshi 包: {e}")
        return False

def process_audio_chunk(audio_data, text_prompt, source_lang, target_lang):
    """处理音频块"""
    try:
        import moshi.offline as moshi_offline
        
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
        
        # Mac M3 内存优化
        # PersonaPlex 7B 模型约需 14-16GB (float16)，加上其他组件约 20GB+
        # Mac M3 48GB 完全足够，但使用 CPU offload 可以优化内存使用
        if device == "mps":
            moshi_device = "cpu"  # Mac 上使用 CPU（MPS 可能不稳定）
            # 检查是否安装了 accelerate（CPU offload 需要）
            try:
                import accelerate
                use_cpu_offload = True
                print("✓ Mac M3 检测到，使用 CPU + CPU offload 模式（已安装 accelerate）")
            except ImportError:
                use_cpu_offload = False
                print("⚠️  Mac M3 检测到，建议安装 accelerate 以优化内存: pip install accelerate")
        else:
            moshi_device = device
            use_cpu_offload = (moshi_device == "cpu")
        
        # 获取 voice prompt 目录
        try:
            voice_prompt_dir = moshi_offline._get_voice_prompt_dir(None, "nvidia/personaplex-7b-v1")
            voice_prompt = os.path.join(voice_prompt_dir, "NATF2.pt")
            if not os.path.exists(voice_prompt):
                voice_prompt = "NATF2.pt"
        except:
            voice_prompt = "NATF2.pt"
        
        # 确保使用 HF_TOKEN（在每次处理时都登录，确保 token 有效）
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if hf_token:
            try:
                from huggingface_hub import login
                # 静默登录，不显示警告
                login(token=hf_token, add_to_git_credential=False)
                print("✓ 已使用 HF_TOKEN 登录 Hugging Face")
            except Exception as e:
                print(f"⚠️  登录 Hugging Face 失败: {e}")
                print("   请检查：")
                print("   1. Token 是否启用了 'Enable access to public gated repositories'")
                print("   2. 是否接受了模型许可: https://huggingface.co/nvidia/personaplex-7b-v1")
        else:
            print("⚠️  警告: 未设置 HF_TOKEN，可能无法下载模型文件")
        
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
        
        # 读取输出音频
        audio_output, sr = librosa.load(output_path, sr=24000)
        
        # 清理临时文件
        os.unlink(input_path)
        os.unlink(output_path)
        os.unlink(text_path)
        
        return audio_output
    except Exception as e:
        print(f"处理错误: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    return send_file('index_realtime_simple.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    mps_available = False
    if hasattr(torch.backends, 'mps'):
        mps_available = torch.backends.mps.is_available()
    
    return jsonify({
        'model_loaded': model_loaded,
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': mps_available
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
    """处理实时音频块 - 简化版本，直接使用二进制数组"""
    try:
        # 调试：打印接收到的数据
        print(f"收到数据，键: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        print(f"数据类型: {type(data)}")
        
        # 接收二进制数组
        audio_array = data.get('audio')
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'zh')
        
        # 调试信息
        if audio_array is None:
            print("⚠️  audio 键不存在")
            print(f"可用的键: {list(data.keys())}")
            return
        
        if not isinstance(audio_array, list):
            print(f"⚠️  audio 不是列表类型，而是: {type(audio_array)}")
            print(f"audio 值的前 10 个元素: {audio_array[:10] if hasattr(audio_array, '__getitem__') else 'N/A'}")
            return
        
        if len(audio_array) == 0:
            print("⚠️  audio 数组为空")
            return
        
        # 转换为 bytes
        audio_bytes = bytes(audio_array)
        print(f"收到音频数据，大小: {len(audio_bytes)} bytes")
        
        # 验证文件头
        if len(audio_bytes) < 4:
            print("音频数据太短")
            return
        
        header = audio_bytes[:4]
        print(f"文件头: {header.hex()} ({header})")
        
        if header != b'RIFF':
            print(f"⚠️  不是有效的 WAV 文件，文件头: {header}")
            return
        
        # 保存为临时文件
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_input.write(audio_bytes)
        temp_input.close()
        temp_path = temp_input.name
        
        try:
            # 加载音频
            audio_data, sr = librosa.load(temp_path, sr=24000)
            print(f"成功加载音频: {len(audio_data)} 采样点")
        except Exception as e:
            print(f"音频加载错误: {e}")
            os.unlink(temp_path)
            return
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # 创建翻译提示词
        lang_names = {
            "en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French",
            "de": "German", "ja": "Japanese", "ko": "Korean"
        }
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        text_prompt = f"""You are a professional real-time translator. Translate speech from {source_name} to {target_name}. Always respond in {target_name} only."""
        
        # 处理音频（在后台线程）
        def process_and_send():
            translated_audio = process_audio_chunk(audio_data, text_prompt, source_lang, target_lang)
            if translated_audio is not None:
                # 转换为 base64
                output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                sf.write(output_temp.name, translated_audio, 24000)
                output_temp.close()
                
                with open(output_temp.name, 'rb') as f:
                    audio_bytes = f.read()
                os.unlink(output_temp.name)
                
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                socketio.emit('translated_audio', {'audio': audio_base64})
        
        threading.Thread(target=process_and_send, daemon=True).start()
        
    except Exception as e:
        print(f"处理音频块错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    print(f"启动服务器在端口 {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)

