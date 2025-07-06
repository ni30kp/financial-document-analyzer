"""
Utility script to create sample PDF from text file
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.colors import black

def create_sample_pdf():
    """Create a sample PDF from the text file"""
    
    # Read the sample text file
    with open("data/sample_financial_report.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Create PDF
    pdf_path = "data/sample_financial_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        textColor=black
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=12,
        textColor=black
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        textColor=black
    )
    
    # Process content
    story = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        
        # Title
        if line.startswith('TECHCORP FINANCIAL REPORT'):
            story.append(Paragraph(line, title_style))
        # Headings
        elif line.isupper() and len(line) > 3:
            story.append(Paragraph(line, heading_style))
        # Normal text
        else:
            story.append(Paragraph(line, normal_style))
    
    # Build PDF
    doc.build(story)
    print(f"Sample PDF created: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    # Install reportlab if not already installed
    try:
        import reportlab
    except ImportError:
        print("Installing reportlab...")
        import subprocess
        subprocess.check_call(["pip", "install", "reportlab"])
    
    create_sample_pdf() 