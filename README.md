# mapIA

Preparação de datasets de mapas para ResNet: lê imagens e PDFs, recorta a moldura dourada e salva em formato `ImageFolder`.

O pipeline está em [`recorte_database/`](recorte_database/). Calibração, flags e limitações estão no [README de lá](recorte_database/README.md).

## Instalação

Na raiz do repositório (Python >= 3.9):

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Uso

```bash
cd recorte_database
python main.py -i data/raw -o data/processed --workers 8
```
