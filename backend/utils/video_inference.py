import cv2
import uuid
import os
from pathlib import Path
from utils.inference import load_model

def run_video_inference(video_path, model_name):
    """Run inference on video and return the path to the processed video"""
    try:
        print(f"🔄 Starting video inference with model: {model_name}")
        model, method = load_model(model_name)
        print(f"✅ Model loaded successfully using method: {method}")

        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video info: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Create output path - use AVI format which is more compatible
        result_path = Path("static/results") / f"video_result_{uuid.uuid4().hex}.avi"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use XVID codec which is more widely supported
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(result_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            # Fallback to MJPG if XVID fails
            print("⚠️ XVID codec failed, trying MJPG...")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(str(result_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise Exception("Could not initialize video writer with XVID or MJPG codecs")
        
        print(f"✅ Using codec: {'XVID' if fourcc == cv2.VideoWriter_fourcc(*'XVID') else 'MJPG'}")
        
        # Process frames
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            try:
                if method == "ultralytics":
                    results = model(frame)
                    result_img = results[0].plot()
                else:
                    # Fallback: show no annotation
                    result_img = frame
                
                # Write frame
                out.write(result_img)
                
                # Progress update every 30 frames
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"📊 Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
                    
            except Exception as e:
                print(f"❌ Error processing frame {frame_count}: {e}")
                # Write original frame if inference fails
                out.write(frame)
        
        # Clean up
        cap.release()
        out.release()
        
        # Verify the output file exists and has content
        if not result_path.exists():
            raise Exception("Output video file was not created")
        
        file_size = result_path.stat().st_size
        if file_size == 0:
            raise Exception("Output video file is empty")
        
        print(f"✅ Video processing complete: {result_path} ({file_size / 1024 / 1024:.1f} MB)")
        
        # Try to create a browser-compatible MP4 version using ffmpeg
        try:
            import subprocess
            mp4_path = result_path.with_suffix('.mp4')
            print(f"🔄 Converting to MP4 for browser compatibility...")
            
            # Use FFmpeg to convert AVI to MP4 with H.264 codec
            cmd = [
                'ffmpeg', '-i', str(result_path), 
                '-c:v', 'libx264', '-preset', 'fast', 
                '-crf', '23', '-movflags', '+faststart',
                str(mp4_path), '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and mp4_path.exists():
                mp4_size = mp4_path.stat().st_size
                print(f"✅ Created browser-compatible MP4: {mp4_path} ({mp4_size / 1024 / 1024:.1f} MB)")
                
                # Remove the AVI file to save space
                try:
                    result_path.unlink()
                    print(f"🗑️ Removed AVI file to save space")
                except:
                    pass
                
                return str(mp4_path)
            else:
                print(f"⚠️ FFmpeg conversion failed: {result.stderr}")
                print(f"📹 Using AVI file: {result_path}")
                return str(result_path)
                
        except subprocess.TimeoutExpired:
            print(f"⚠️ FFmpeg conversion timed out, using AVI file")
            return str(result_path)
        except FileNotFoundError:
            print(f"⚠️ FFmpeg not found, using AVI file")
            return str(result_path)
        except Exception as e:
            print(f"⚠️ FFmpeg conversion failed: {e}")
            print(f"📹 Using AVI file: {result_path}")
            return str(result_path)
        
    except Exception as e:
        print(f"❌ Video inference failed: {e}")
        # Clean up any partial files
        if 'result_path' in locals() and result_path.exists():
            try:
                result_path.unlink()
            except:
                pass
        raise e
