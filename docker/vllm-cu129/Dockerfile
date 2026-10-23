# vllm/vllm-openai:v0.32.0-cu129 ships a CUDA 13 torchcodec (needs libnvrtc.so.13) and crashes on import.
# torchcodec is only used for CPU audio/video decoding, so swap in the CPU build.
FROM vllm/vllm-openai:v0.32.0-cu129
RUN pip install --no-cache-dir --force-reinstall --no-deps "torchcodec==0.17.0" --index-url https://download.pytorch.org/whl/cpu
