import os
import shutil
import pymupdf


def extrair_com_pymupdf(pdf_path: str, output_dir: str) -> None:
  if not os.path.exists(pdf_path):
    print(f"Erro: Arquivo PDF não encontrado no caminho:\n{pdf_path}")
    return

  os.makedirs(output_dir, exist_ok=True)
  doc = pymupdf.open(pdf_path)

  xrefs_extraidos = set()
  total_salvas = 0

  for page_index in range(len(doc)):
    page = doc[page_index]
    for img_info in page.get_images(full=True):
      xref = img_info[0]

      if xref in xrefs_extraidos:
        continue
      xrefs_extraidos.add(xref)

      base_image = doc.extract_image(xref)
      image_bytes = base_image["image"]
      image_ext = base_image["ext"]

      nome_pdf = os.path.splitext(os.path.basename(pdf_path))[0]
      nome_saida = f"{nome_pdf}_pag_{page_index + 1:03d}_xref_{xref}.{image_ext}"
      caminho_completo = os.path.join(output_dir, nome_saida)

      with open(caminho_completo, "wb") as f:
        f.write(image_bytes)

      total_salvas += 1

  doc.close()
  print(f"Sucesso! {total_salvas} imagens originais salvas em:\n{output_dir}")


raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
origem_raiz = os.path.join(raiz, "data", "raw", "imagens", "Albernaz Velho_Exemplos")
destino_raiz = os.path.join(raiz, "rafa_scripts", "output", "Albernaz Velho_Exemplos")

for nome_pasta in sorted(os.listdir(origem_raiz)):
  pasta_origem = os.path.join(origem_raiz, nome_pasta)
  if not os.path.isdir(pasta_origem):
    continue

  pasta_saida = os.path.join(destino_raiz, nome_pasta)

  for nome_arquivo in sorted(os.listdir(pasta_origem)):
    caminho = os.path.join(pasta_origem, nome_arquivo)
    if not os.path.isfile(caminho):
      continue

    if nome_arquivo.lower().endswith(".pdf"):
      extrair_com_pymupdf(caminho, pasta_saida)
    else:
      os.makedirs(pasta_saida, exist_ok=True)
      shutil.copy2(caminho, os.path.join(pasta_saida, nome_arquivo))
    #   print(f"Copiado: {nome_arquivo}\n{pasta_saida}")
