# mapIA

Piloto de 21 dias (19/11 a 10/12): curadoria manual de cartas de Albernaz I e, na semana de 26/11, extração de vetores com ResNet-50 pré-treinada e similaridade cosseno (carta inteira e recortes). A carta de Pernambuco (ca. 1630) fica isolada como consulta.

A estrutura e as tabelas estão em [`piloto_albernaz/`](piloto_albernaz/). Instruções da semana em [`piloto_albernaz/LEIA-ME.md`](piloto_albernaz/LEIA-ME.md).

O pipeline automático de recorte em [`recorte_database/`](recorte_database/) permanece no repositório e fica fora deste prazo.

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
