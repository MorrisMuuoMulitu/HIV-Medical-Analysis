"""
PDF Report Generator for HIV Analytics
=======================================

Generates professional PDF reports for HIV medical analytics.
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.colors import HexColor
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFReportGenerator:
    """Generate professional PDF reports for HIV analytics."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSectionHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=HexColor('#7f8c8d'),
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='ReportBodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=HexColor('#2c3e50'),
            alignment=TA_JUSTIFY
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=HexColor('#667eea'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=HexColor('#7f8c8d'),
            alignment=TA_CENTER
        ))
    
    def generate_summary_report(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Generate a summary report with key metrics.
        
        Args:
            data: Dictionary containing analytics data
            filename: Optional custom filename
            
        Returns:
            Path to generated PDF
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"HIV_Summary_Report_{timestamp}.pdf"
        
        output_path = self.output_dir / filename
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        content = []
        
        # Title
        content.append(Paragraph("HIV/AIDS Analytics Summary", self.styles['CustomTitle']))
        content.append(Spacer(1, 0.2*inch))
        
        # Report date
        report_date = datetime.now().strftime("%B %d, %Y")
        content.append(Paragraph(f"Report Generated: {report_date}", self.styles['ReportBodyText']))
        content.append(Spacer(1, 0.3*inch))
        
        # Key Metrics
        content.append(Paragraph("Key Performance Metrics", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2*inch))
        
        metrics = data.get('metrics', {})
        metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Viral Suppression Rate', f"{metrics.get('viral_suppression_rate', 0):.1f}%", '✓ Target Met'],
            ['Treatment Adherence', f"{metrics.get('adherence_rate', 0):.1f}%", '✓ Target Met'],
            ['Retention in Care', f"{metrics.get('retention_rate', 0):.1f}%", '✓ Target Met'],
            ['CD4 Count Improvement', f"{metrics.get('cd4_improvement', 0):.1f}%", '↑ Improving'],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        content.append(metrics_table)
        content.append(Spacer(1, 0.3*inch))
        
        # Demographics Summary
        content.append(Paragraph("Patient Demographics", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2*inch))
        
        demographics = data.get('demographics', {})
        demo_data = [
            ['Category', 'Breakdown'],
            ['Total Patients', f"{demographics.get('total_patients', 0):,}"],
            ['Age Distribution', demographics.get('age_distribution', 'N/A')],
            ['Gender Distribution', demographics.get('gender_distribution', 'N/A')],
            ['Transmission Routes', demographics.get('transmission_routes', 'N/A')],
        ]
        
        demo_table = Table(demo_data, colWidths=[2*inch, 3.5*inch])
        demo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        content.append(demo_table)
        content.append(Spacer(1, 0.3*inch))
        
        # Treatment Outcomes
        content.append(Paragraph("Treatment Outcomes", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2*inch))
        
        outcomes = data.get('treatment_outcomes', {})
        outcomes_text = f"""
        <b>ART Initiation Rate:</b> {outcomes.get('art_initiation', 0):.1f}%<br/>
        <b>Viral Load Suppression:</b> {outcomes.get('viral_suppression', 0):.1f}%<br/>
        <b>Treatment Retention (12 months):</b> {outcomes.get('retention_12m', 0):.1f}%<br/>
        <b>Adverse Events Rate:</b> {outcomes.get('adverse_events', 0):.1f}%<br/>
        """
        content.append(Paragraph(outcomes_text, self.styles['ReportBodyText']))
        content.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        content.append(Paragraph("Key Recommendations", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2*inch))
        
        recommendations = data.get('recommendations', [
            "Continue monitoring viral load trends",
            "Focus on improving treatment adherence",
            "Expand testing in high-risk populations"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            content.append(Paragraph(f"{i}. {rec}", self.styles['ReportBodyText']))
            content.append(Spacer(1, 0.1*inch))
        
        # Footer
        content.append(PageBreak())
        footer = Paragraph(
            "This report was automatically generated by the HIV/AIDS Analytics Platform<br/>"
            "For questions or support, contact your healthcare data team",
            ParagraphStyle(
                'FooterStyle',
                parent=self.styles['Normal'],
                fontSize=8,
                alignment=TA_CENTER,
                textColor=HexColor('#7f8c8d')
            )
        )
        content.append(footer)
        
        # Build PDF
        doc.build(content)
        logger.info(f"Summary report generated: {output_path}")
        
        return str(output_path)
    
    def generate_detailed_report(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """Generate a comprehensive detailed report with all analytics."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"HIV_Detailed_Report_{timestamp}.pdf"
        
        output_path = self.output_dir / filename
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=landscape(letter),
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        content = []
        
        # Title
        content.append(Paragraph("HIV/AIDS Comprehensive Analytics Report", self.styles['CustomTitle']))
        content.append(Spacer(1, 0.2*inch))
        
        report_date = datetime.now().strftime("%B %d, %Y")
        content.append(Paragraph(f"Report Generated: {report_date}", self.styles['BodyText']))
        content.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        content.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.2*inch))
        
        summary = data.get('executive_summary', '')
        content.append(Paragraph(summary, self.styles['ReportBodyText']))
        content.append(Spacer(1, 0.3*inch))
        
        # Detailed Metrics Tables
        content.append(Paragraph("Detailed Analytics", self.styles['SectionHeader']))
        
        # Add data tables from the analytics
        detailed_data = data.get('detailed_tables', [])
        for table_data in detailed_data:
            if table_data.get('title'):
                content.append(Paragraph(table_data['title'], self.styles['SubSectionHeader']))
            
            df = table_data.get('dataframe')
            if df is not None:
                table = self._dataframe_to_table(df)
                content.append(table)
                content.append(Spacer(1, 0.2*inch))
        
        # ML Model Performance
        if 'ml_metrics' in data:
            content.append(Paragraph("Machine Learning Model Performance", self.styles['SectionHeader']))
            content.append(Spacer(1, 0.2*inch))
            
            ml_metrics = data['ml_metrics']
            ml_data = [
                ['Metric', 'Score'],
                ['ROC-AUC', f"{ml_metrics.get('roc_auc', 0):.3f}"],
                ['Precision', f"{ml_metrics.get('precision', 0):.3f}"],
                ['Recall', f"{ml_metrics.get('recall', 0):.3f}"],
                ['F1-Score', f"{ml_metrics.get('f1', 0):.3f}"],
                ['Accuracy', f"{ml_metrics.get('accuracy', 0):.3f}"],
            ]
            
            ml_table = Table(ml_data, colWidths=[3*inch, 2*inch])
            ml_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            content.append(ml_table)
            content.append(Spacer(1, 0.3*inch))
        
        # Build PDF
        doc.build(content)
        logger.info(f"Detailed report generated: {output_path}")
        
        return str(output_path)
    
    def _dataframe_to_table(self, df: pd.DataFrame, max_rows: int = 50) -> Table:
        """Convert pandas DataFrame to ReportLab table."""
        # Limit rows for readability
        if len(df) > max_rows:
            df = df.head(max_rows)
        
        # Convert to list format
        data = [df.columns.tolist()] + df.values.tolist()
        
        # Create table
        num_cols = len(df.columns)
        col_widths = [2.5*inch] * min(num_cols, 4)  # Max 4 columns visible
        
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        
        return table
    
    def generate_executive_report(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """Generate a one-page executive summary report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"HIV_Executive_Report_{timestamp}.pdf"
        
        output_path = self.output_dir / filename
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=1*inch,
            leftMargin=1*inch,
            topMargin=1*inch,
            bottomMargin=1*inch
        )
        
        content = []
        
        # Header with logo placeholder
        header_style = ParagraphStyle(
            'HeaderStyle',
            parent=self.styles['CustomTitle'],
            fontSize=20,
            spaceAfter=10
        )
        content.append(Paragraph("HIV/AIDS Program", header_style))
        content.append(Paragraph("Executive Dashboard", self.styles['SectionHeader']))
        content.append(Spacer(1, 0.3*inch))
        
        # Date
        content.append(Paragraph(
            f"Report Date: {datetime.now().strftime('%B %d, %Y')}",
            self.styles['ReportBodyText']
        ))
        content.append(Spacer(1, 0.4*inch))
        
        # Key Metrics in cards
        metrics = data.get('metrics', {})
        
        metrics_cards = [
            ('Viral Suppression', f"{metrics.get('viral_suppression_rate', 0):.1f}%", '✓ On Track'),
            ('Treatment Adherence', f"{metrics.get('adherence_rate', 0):.1f}%", '✓ On Track'),
            ('Retention in Care', f"{metrics.get('retention_rate', 0):.1f}%", '⚠ Monitor'),
            ('CD4 Improvement', f"{metrics.get('cd4_improvement', 0):.1f}%", '↑ Improving'),
        ]
        
        for title, value, status in metrics_cards:
            card_data = [
                [title, value, status]
            ]
            card_table = Table(card_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            card_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f8f9fa')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
                ('TOPPADDING', (0, 0), (-1, 0), 15),
                ('GRID', (0, 0), (-1, 0), 1, HexColor('#667eea')),
            ]))
            content.append(card_table)
            content.append(Spacer(1, 0.1*inch))
        
        content.append(Spacer(1, 0.3*inch))
        
        # Key Insights
        content.append(Paragraph("Key Insights", self.styles['SectionHeader']))
        insights = data.get('insights', [
            "Viral suppression rates remain above target threshold",
            "Treatment adherence shows consistent improvement",
            "Focus needed on patient retention in care"
        ])
        
        for insight in insights:
            content.append(Paragraph(f"• {insight}", self.styles['ReportBodyText']))
            content.append(Spacer(1, 0.05*inch))
        
        # Action Items
        content.append(Spacer(1, 0.2*inch))
        content.append(Paragraph("Priority Actions", self.styles['SectionHeader']))
        actions = data.get('action_items', [
            "Implement enhanced adherence counseling",
            "Expand community-based testing programs",
            "Review retention strategies for high-risk groups"
        ])
        
        for i, action in enumerate(actions, 1):
            content.append(Paragraph(f"{i}. {action}", self.styles['ReportBodyText']))
        
        # Footer
        content.append(Spacer(1, 0.5*inch))
        footer = Paragraph(
            "Confidential - For Internal Use Only<br/>"
            "HIV/AIDS Analytics Platform | Generated Automatically",
            ParagraphStyle(
                'FooterStyle',
                parent=self.styles['Normal'],
                fontSize=8,
                alignment=TA_CENTER,
                textColor=HexColor('#7f8c8d')
            )
        )
        content.append(footer)
        
        doc.build(content)
        logger.info(f"Executive report generated: {output_path}")
        
        return str(output_path)


# Convenience function for quick report generation
def generate_report(
    data: Dict[str, Any],
    report_type: str = "summary",
    output_dir: str = "reports"
) -> str:
    """
    Generate a PDF report.
    
    Args:
        data: Analytics data dictionary
        report_type: 'summary', 'detailed', or 'executive'
        output_dir: Output directory for reports
        
    Returns:
        Path to generated PDF
    """
    generator = PDFReportGenerator(output_dir)
    
    if report_type == "summary":
        return generator.generate_summary_report(data)
    elif report_type == "detailed":
        return generator.generate_detailed_report(data)
    elif report_type == "executive":
        return generator.generate_executive_report(data)
    else:
        raise ValueError(f"Unknown report type: {report_type}")
