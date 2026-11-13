import os
import numpy as np
from PIL import Image

# Evita erro de DecompressionBombError em mapas gigantescos (ex: TIFs de alta resolução)
Image.MAX_IMAGE_PIXELS = None


def gerar_patches_obra(
    pasta_obra: str,
    pasta_destino_patches: str,
    tamanho_patch: int = 224,
    stride: int = 224,
    min_resolucao_img: int = 800,
    min_desvio_padrao: float = 18.0,
) -> int:
  """Fatia mapas de uma obra em patches RGB, filtrando lixo de PDF e áreas vazias."""
  os.makedirs(pasta_destino_patches, exist_ok=True)
  extensoes_validas = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2")
  total_patches = 0

  for arquivo in sorted(os.listdir(pasta_obra)):
    if not arquivo.lower().endswith(extensoes_validas):
      continue

    caminho_img = os.path.join(pasta_obra, arquivo)
    try:
      with Image.open(caminho_img) as img:
        largura, altura = img.size

        # 1. Filtra ícones, selos ou miniaturas extraídas do PDF
        if largura < min_resolucao_img or altura < min_resolucao_img:
          continue

        # Converte CMYK / Grayscale / RGBA para RGB padrão da ResNet
        img_rgb = img.convert("RGB")
        nome_base = os.path.splitext(arquivo)[0]

        for y in range(0, altura - tamanho_patch + 1, stride):
          for x in range(0, largura - tamanho_patch + 1, stride):
            patch = img_rgb.crop((x, y, x + tamanho_patch, y + tamanho_patch))
            arr = np.asarray(patch, dtype=np.float32)

            # 2. Filtra patches que caíram em papel em branco / margem vazia
            if np.std(arr) < min_desvio_padrao:
              continue

            nome_patch = f"{nome_base}_y{y:04d}_x{x:04d}.jpg"
            patch.save(
                os.path.join(pasta_destino_patches, nome_patch),
                "JPEG",
                quality=95,
            )
            total_patches += 1

    except Exception as e:
      print(f"Aviso: falha ao processar {caminho_img}: {e}")

  return total_patches


raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
origem_obras = os.path.join(
    raiz, "rafa_scripts", "output", "Albernaz Velho_Exemplos"
)
destino_patches = os.path.join(
    raiz, "rafa_scripts", "output", "patches_albernaz"
)

for nome_obra in sorted(os.listdir(origem_obras)):
  caminho_obra = os.path.join(origem_obras, nome_obra)
  if not os.path.isdir(caminho_obra):
    continue

  qtd = gerar_patches_obra(
      pasta_obra=caminho_obra,
      pasta_destino_patches=os.path.join(destino_patches, nome_obra),
      tamanho_patch=224,
      stride=224,  # Use stride=112 se quiser sobreposição (50%) para aumentar o dataset
  )
  print(f"[{nome_obra}] -> {qtd} patches úteis gerados.")
