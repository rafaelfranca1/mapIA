# Preparação de dataset de mapas para ResNet

Lê `.tiff/.tif/.jpg/.jpeg/.png/.pdf` recursivamente, converte para RGB uint8, **recorta da moldura
dourada para dentro**, padroniza com **letterbox** (sem distorção) e salva em estrutura compatível com
`torchvision.datasets.ImageFolder`.

## Instalação
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # Python >= 3.9; sem Poppler
```

## Uso
```bash
# 1) CALIBRAR (não grava o dataset): 30 painéis  Antes | Máscara dourada | Depois
python main.py -i data/raw --dry-run --debug-samples 30 --debug-dir data/debug

# 2) Rodar
python main.py -i data/raw -o data/processed --workers 8

# Variações
python main.py -i data/raw -o out --size 1024 --format jpg --quality 95
python main.py -i data/raw -o out --frame-fallback skip       # descarta o que não tem moldura
python main.py -i data/raw -o out --crop-mode white           # só remove borda branca
python main.py -i data/raw -o out --workers 1 -v              # sequencial + logs detalhados
```

## Resolução e formato
- A perda de resolução vem do **tamanho final** (`--size`), não do formato. Um mapa de 6000 px reduzido para 224 perde
  letras e traços finos. O padrão agora é **512**; grave em resolução maior do que vai treinar (não dá para recuperar depois)
  e reduza no treino (`RandomResizedCrop(224)`). A ResNet aceita 384/512 (custo ~5x em 512 vs 224).
- Padrão de saída: **JPEG q95, 4:4:4** (muito menor que PNG; diferença visual desprezível). Use `--format png` se quiser sem perdas.

## Recorte pela moldura dourada
1. Remove a borda branca externa. 2. Máscara de "dourado" em HSV. 3. Em cada lado, acha a banda dourada colada à borda
e recorta na borda **interna** dela (+ respiro `--frame-inner-margin`, padrão 0,2%).
Se a moldura não for achada (ou o recorte for implausível), aplica `--frame-fallback`: `keep` (borda branca + aviso no log
e contagem no relatório) ou `skip` (descarta e conta como "rejeitada").

**Calibração** (olhe `data/debug`; azul = borda branca, vermelho = recorte final, painel central = máscara dourada):

| Sintoma | Ajuste |
|---|---|
| Máscara pega o miolo (pergaminho amarelado) | subir `--frame-min-sat` (ex.: 120) ou estreitar `--frame-hue` |
| Moldura não aparece na máscara | descer `--frame-min-sat`/`--frame-min-val` ou alargar `--frame-hue` (ex.: `10,40`) |
| Moldura detectada em 3/4 lados | `--frame-min-sides 3` ou baixar `--frame-band-ratio` (0.4) |
| Sobra um fio dourado dentro | subir `--frame-inner-margin` (ex.: `0.005`) |
| Molduras muito grossas | subir `--frame-search` (padrão 0.25 = 25% do lado) |

## Arquitetura (OO + polimorfismo)
```
main.py                      CLI -> Settings -> DatasetPipeline
config.py                    constantes + Settings (dataclass imutável)
dataset_prep/
  readers.py    ImageReader (ABC) -> PdfReader | TiffReader | RasterReader ; ReaderRegistry (despacho por extensão)
  croppers.py   Cropper (ABC)     -> WhiteBorderCropper | GoldFrameCropper (decora um Cropper base) | NullCropper
  resizers.py   Resizer (ABC)     -> LetterboxResizer
  writers.py    ImageWriter (ABC) -> PngWriter | JpegWriter
  processing.py ImageProcessor (Cropper + Resizer injetados) ; FileProcessor (leitura -> processamento -> escrita)
  pipeline.py   DatasetPipeline: descoberta, multiprocessing, relatório
  factory.py    Settings -> objetos concretos (CROPPER_BUILDERS / WRITER_BUILDERS)
  reporting.py  logging + Report ;  debug.py DebugRenderer ;  models.py / geometry.py / conversions.py
```
Para estender, implemente a ABC e registre em `factory.py`. Ex.: novo formato = subclasse de `ImageReader` na lista de
`build_readers`; novo recorte = subclasse de `Cropper` + entrada em `CROPPER_BUILDERS` (e em `config.CROP_MODES`).

## Comportamento
- Hierarquia de pastas preservada. Multipágina: `nome_page_1`, `nome_page_2`... Colisões (`a.jpg` + `a.png`) viram `a_jpg`, `a_png`.
- PDFs grandes (plantas): lado máx. `--pdf-max-side` (8000 px); DPI reduzido automaticamente.
- Falhas por arquivo (corrompido, PDF com senha) são registradas em `processing.log` e não param o pipeline.
- Relatório: por extensão, geradas, fallback, vazias, rejeitadas, erros, por classe, tempo.

## Limitações
- A moldura é detectada por **cor**; pergaminho muito saturado ou moldura desbotada exigem calibração.
- Esperado moldura alinhada aos eixos (inclinação de ~1° é tolerada); mapas muito rotacionados não são endireitados.
- 16-bit→8-bit linear (TIFF de 12 bits em 16 pode escurecer); CMYK→RGB sem perfil ICC.
