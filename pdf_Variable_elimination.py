from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import txt2pdf
import PyPDF2
# Ruta del archivo PDF
class args():
    def __init__(self):
        self.filename='archivo_salida.txt'
        self.font='Helvetica'

        self.font_size=9.0
        self.extra_vertical_space=0.0
        self.kerning=0.0
        self.media='A4'
        self.landscape=False
        self.margin_left=2.0
        self.margin_right=2.0
        self.margin_top=2.0
        self.margin_bottom=2.0
        self.output='output_VE.pdf'
        self.author=''
        self.title=''
        self.quiet=False
        self.break_on_blanks=False
        self.encoding='utf8'
        self.page_numbers=False
        self.line_numbers=False
    
def generarPDF():
    ruta_archivo = 'informe_VE.pdf'
    w, h = A4
    x = 120
    y = h - 45
    # Crear un lienzo para el PDF
    c = canvas.Canvas(ruta_archivo, pagesize=letter)

    # Título del informe
    titulo = "Variable Elimination - reasoning"
    titulo2='aaa'
    c.setFont("Helvetica-Bold", 15)
    c.drawString(200, h-80, titulo2)

    c.drawString(200, h-100, 'hola')

   
    ruta_imagen1 = 'reduced.png'
    c.drawImage(ruta_imagen1,80, h-460, width=340, height=340)
    
    c.showPage()
    ruta_imagen2 = 'Cost_variable_eliminationordering.png'
    c.drawImage(ruta_imagen2,80, h-80, width=340, height=340)

   
    c.save()
    txt2pdf.PDFCreator(args(),txt2pdf.Margins(right=2.0, left=2.0, top=2.0, bottom=2.0)).generate()
    




    # Abrir los archivos PDF en modo de lectura binaria
    with open('informe_VE.pdf', "rb") as pdf1_file, open('output_VE.pdf', "rb") as pdf2_file:
        # Crear objetos PDFReader para los archivos PDF
        pdf1_reader = PyPDF2.PdfReader(pdf1_file)
        pdf2_reader = PyPDF2.PdfReader(pdf2_file)

        # Crear un nuevo objeto PDFWriter
        pdf_writer = PyPDF2.PdfWriter()

        # Agregar todas las páginas del archivo 1 al PDFWriter
        for page_num in range(len(pdf1_reader.pages)):
            page = pdf1_reader.pages[page_num]
            pdf_writer.add_page(page)

        # Agregar todas las páginas del archivo 2 al PDFWriter
        for page_num in range(len(pdf2_reader.pages)):
            page = pdf2_reader.pages[page_num]
            pdf_writer.add_page(page)

        # Guardar el PDF concatenado en un nuevo archivo
        output_file = "archivo_concatenado_VE.pdf"
        with open(output_file, "wb") as output:
            pdf_writer.write(output)

        print("La concatenación se ha completado. El archivo resultante se encuentra en:", output_file)
