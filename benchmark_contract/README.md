# benchmark_contract for OmniFlowNet

Adaptação local do contrato experimental para o workspace real do OmniFlowNet.

## Caminho recomendado
Use o container Linux/GPU em [benchmark_contract/Dockerfile.benchmark](/Volumes/External%20SSD/Mestrado/BenchmarkIndividualOF/OmniFlowNet/benchmark_contract/Dockerfile.benchmark). Ele:
- clona o repositório oficial `twhui/LiteFlowNet`;
- aplica o patch OmniFlowNet em `im2col.cu`;
- compila `caffe.bin` dentro da imagem;
- instala um Python moderno separado para o contrato;
- usa `caffe.bin test` para gerar o `.flo` real quando um `.caffemodel` estiver disponível.

## Checkpoint
Há dois jeitos suportados para delegar o `.caffemodel` ao build:
1. copiar o arquivo para `benchmark_contract/docker_assets/checkpoints/` antes do build;
2. passar `--build-arg OMNIFLOWNET_CAFFEMODEL_URL=...` com um link direto para download.

Se nenhum checkpoint existir, o contrato ainda roda, mas registra explicitamente execução degradada.

## Build
```bash
docker build -f benchmark_contract/Dockerfile.benchmark -t omniflownet-bench .
```

## Run
Monte o dataset no mesmo caminho esperado pelo projeto:
```bash
docker run --rm --gpus all \
  -v /host/path/OMNIFLOWNET_DATASET:/app/OMNIFLOWNET_DATASET:ro \
  -v "$(pwd)/benchmark_contract/results:/app/benchmark_contract/results" \
  -v "$(pwd)/benchmark_contract/outputs:/app/benchmark_contract/outputs" \
  omniflownet-bench \
  official_reproduction
```

## Saídas
Os arquivos JSON são escritos em `benchmark_contract/results/`.
