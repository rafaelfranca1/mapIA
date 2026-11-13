"""Pipeline OO de preparação de datasets de imagens para ResNet (ImageFolder).

Camadas (cada uma com uma ABC e implementações intercambiáveis):
    readers   -> ImageReader   (PdfReader, TiffReader, RasterReader)
    croppers  -> Cropper       (WhiteBorderCropper, GoldFrameCropper, InkFrameCropper, NullCropper)
    resizers  -> Resizer       (LetterboxResizer)
    writers   -> ImageWriter   (PngWriter, JpegWriter)
"""
