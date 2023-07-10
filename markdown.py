from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import txt2pdf
import PyPDF2
# Ruta del archivo PDF
class args():
    def __init__(self):
        self.filename='Comparison_evidence.txt'
        self.font='Helvetica'
        self.font_size=10.0
        self.extra_vertical_space=0.0
        self.kerning=0.0
        self.media='A4'
        self.landscape=False
        self.margin_left=2.0
        self.margin_right=2.0
        self.margin_top=2.0
        self.margin_bottom=2.0
        self.output='output.pdf'
        self.author=''
        self.title=''
        self.quiet=False
        self.break_on_blanks=False
        self.encoding='utf8'
        self.page_numbers=False
        self.line_numbers=False
    
def generarPDF():
    ruta_archivo = 'informe_JT.pdf'
    w, h = A4
    x = 120
    y = h - 45
    # Crear un lienzo para el PDF
    c = canvas.Canvas(ruta_archivo, pagesize=letter)

    # Título del informe
    titulo = "Junction tree - reasoning"
    c.setFont("Helvetica-Bold", 15)
    c.drawString(200, h-80, titulo)

    c.setFont('Helvetica-Bold', 14)
    resultado = "Sequence of trees"
    c.drawString(60,h-100, resultado)
    c.setFont('Helvetica', 12)
   
    ruta_imagen1 = 'Construccion_junction_tree.jpg'
    c.drawImage(ruta_imagen1,80, h-580, width=460, height=460)

    c.showPage()
    c.setFont('Helvetica-Bold', 14)
    resultado = "Calibration"
    c.drawString(60, h-80, resultado)
    c.setFont('Helvetica', 12)
    linea1= "It is based on message passing that consists of an upward pass and a downward pass. "
    linea2='In a calibrated clique-tree, the marginal probability over particular variables does not depend'
    linea3='on the clique we selected.'
    # linea3=
    c.drawString(60, h-100, linea1)
    c.drawString(60, h-120, linea2)
    c.drawString(60, h-140, linea3)
    # c.drawString(60, h-520, linea3)

    ruta_imagen2 = 'calibration.png'
    c.drawImage(ruta_imagen2, 80,h-440 , width=280, height=280)
   
    c.setFont('Helvetica-Bold', 14)
    resultado3 = "Calibration - low interpretability"
    c.drawString(60, h-460, resultado3)
    ruta_imagen3 = 'calibration(low).png'
    c.drawImage(ruta_imagen3, 80, h-760, width=280, height=280)

    c.showPage()
    c.setFont('Helvetica-Bold', 14)
    resultado = "Subtree"
    c.drawString(60, h-80, resultado)
    ruta_imagen3 = 'subtree0.png'
    c.drawImage(ruta_imagen3, 80, h-450, width=350, height=350)
    c.showPage()
    c.save()
    txt2pdf.PDFCreator(args(),txt2pdf.Margins(right=2.0, left=2.0, top=2.0, bottom=2.0)).generate()
    


