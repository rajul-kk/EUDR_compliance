"""
Due Diligence Statement (DDS) exporter for EUDR compliance.

Converts the deforestation detection report into the structured formats required
by EU Regulation 2023/1115.  Supports three output formats:

  JSON  — matches the EU Information System (EU IS) API schema
  XML   — W3C-valid envelope accepted by TRACES NT
  PDF   — human-readable summary for operators and auditors

Schema fields follow the public EUDR DDS specification:
  https://environment.ec.europa.eu/topics/forests/deforestation/
  regulation-deforestation-free-products_en

Usage:
    from src.dds_exporter import DDSExporter, OperatorInfo, CommodityInfo
    exporter = DDSExporter(operator, commodity)
    records  = exporter.from_report(report_df, farms_csv, evidence_hash)
    exporter.to_json(records, "reports/dds.json")
    exporter.to_xml(records, "reports/dds.xml")
    exporter.to_pdf(records, "reports/dds.pdf")
"""
from __future__ import annotations

import json
import os
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
import logging
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

import pandas as pd

# reportlab is an optional dep — PDF export degrades gracefully if absent
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    _REPORTLAB = True
except ImportError:
    _REPORTLAB = False


# ---------------------------------------------------------------------------
# Input metadata
# ---------------------------------------------------------------------------

@dataclass
class OperatorInfo:
    name: str
    address: str
    country_iso2: str
    eori: str = ""          # Economic Operators Registration and Identification
    email: str = ""
    phone: str = ""


@dataclass
class CommodityInfo:
    hs_code: str            # e.g. "1801" for cocoa beans
    description: str        # e.g. "Cocoa beans, whole or broken, raw or roasted"
    quantity: float
    unit: str               # e.g. "kg", "t", "m3"
    production_start: str   # ISO date, e.g. "2024-01-01"
    production_end: str     # ISO date, e.g. "2024-12-31"


# ---------------------------------------------------------------------------
# DDS record (one per farm)
# ---------------------------------------------------------------------------

@dataclass
class DDSRecord:
    dds_reference: str
    submission_date: str
    operator: OperatorInfo
    commodity: CommodityInfo
    farm_id: str
    country_of_production: str
    latitude: float
    longitude: float
    area_ha: float
    assessment_date: str
    baseline_year: int
    assessment_year: int
    model_version: str
    risk_level: str                  # COMPLIANT | WARNING | VIOLATION
    deforestation_pct: float
    evidence_hash: str               # sha256 of input imagery / audit entry
    notes: str = ""


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class DDSExporter:
    def __init__(
        self,
        operator: OperatorInfo,
        commodity: CommodityInfo,
        model_version: str = "0.1.0",
        baseline_year: int = 2020,
        assessment_year: int = 2024,
    ) -> None:
        self.operator = operator
        self.commodity = commodity
        self.model_version = model_version
        self.baseline_year = baseline_year
        self.assessment_year = assessment_year

    # ------------------------------------------------------------------
    # Build records from pipeline output
    # ------------------------------------------------------------------

    def from_report(
        self,
        report_df: pd.DataFrame,
        farms_csv: str,
        evidence_hash: str = "",
    ) -> List[DDSRecord]:
        """
        Merge deforestation report with farm metadata to produce DDS records.

        Args:
            report_df:     Output of batch_detect_deforestation (one row per farm).
            farms_csv:     Path to inputs/farms_osm.csv for lat/lon/country lookup.
            evidence_hash: sha256 hash from the AuditEntry for this run.

        Returns:
            List of DDSRecord, one per farm.
        """
        farms = pd.read_csv(farms_csv) if os.path.exists(farms_csv) else pd.DataFrame()
        today = datetime.now(timezone.utc).date().isoformat()
        records: List[DDSRecord] = []

        for _, row in report_df.iterrows():
            farm_id = str(row.get("farm_id", ""))

            # Look up farm metadata
            lat, lon, country, area_ha = 0.0, 0.0, self.operator.country_iso2, 0.0
            if not farms.empty:
                meta = farms[farms["farm_id"].str.endswith(farm_id)]
                if not meta.empty:
                    m = meta.iloc[0]
                    lat = float(m.get("lat", 0))
                    lon = float(m.get("lon", 0))
                    country = str(m.get("country_iso2", self.operator.country_iso2))

            risk = str(row.get("alert_level", "UNKNOWN"))
            defor_pct = float(row.get("deforestation_percent", 0.0))

            if risk == "COMPLIANT":
                notes = "No significant deforestation detected since baseline year."
            elif risk == "WARNING":
                notes = (
                    f"Moderate deforestation detected ({defor_pct:.1f}%). "
                    "Further due diligence required before market placement."
                )
            elif risk == "VIOLATION":
                notes = (
                    f"Significant deforestation detected ({defor_pct:.1f}%). "
                    "Product cannot be placed on the EU market under Regulation 2023/1115."
                )
            else:
                notes = risk

            records.append(
                DDSRecord(
                    dds_reference=f"DDS-{uuid.uuid4().hex[:8].upper()}",
                    submission_date=today,
                    operator=self.operator,
                    commodity=self.commodity,
                    farm_id=farm_id,
                    country_of_production=country,
                    latitude=lat,
                    longitude=lon,
                    area_ha=area_ha,
                    assessment_date=today,
                    baseline_year=self.baseline_year,
                    assessment_year=self.assessment_year,
                    model_version=self.model_version,
                    risk_level=risk,
                    deforestation_pct=defor_pct,
                    evidence_hash=evidence_hash,
                    notes=notes,
                )
            )

        return records

    # ------------------------------------------------------------------
    # JSON export  (EU IS API format)
    # ------------------------------------------------------------------

    def to_json(self, records: List[DDSRecord], output_path: str) -> str:
        """Export DDS records to EU IS-compatible JSON."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        def _rec(r: DDSRecord) -> Dict:
            return {
                "ddsReference": r.dds_reference,
                "submissionDate": r.submission_date,
                "regulationReference": "EU 2023/1115",
                "operator": {
                    "name": r.operator.name,
                    "address": r.operator.address,
                    "countryCode": r.operator.country_iso2,
                    "eori": r.operator.eori,
                    "email": r.operator.email,
                    "phone": r.operator.phone,
                },
                "commodity": {
                    "hsCode": r.commodity.hs_code,
                    "description": r.commodity.description,
                    "quantity": r.commodity.quantity,
                    "unit": r.commodity.unit,
                    "productionPeriod": {
                        "start": r.commodity.production_start,
                        "end": r.commodity.production_end,
                    },
                },
                "geolocation": {
                    "farmId": r.farm_id,
                    "countryOfProduction": r.country_of_production,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "areaHa": r.area_ha,
                },
                "riskAssessment": {
                    "assessmentDate": r.assessment_date,
                    "baselineYear": r.baseline_year,
                    "assessmentYear": r.assessment_year,
                    "modelVersion": r.model_version,
                    "conclusion": r.risk_level,
                    "deforestationPercent": r.deforestation_pct,
                    "evidenceHash": r.evidence_hash,
                    "notes": r.notes,
                },
            }

        payload = {
            "schemaVersion": "1.0",
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "totalRecords": len(records),
            "statements": [_rec(r) for r in records],
        }

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        logger.info("DDS JSON written to %s (%d records)", output_path, len(records))
        return output_path

    # ------------------------------------------------------------------
    # XML export  (TRACES NT envelope)
    # ------------------------------------------------------------------

    def to_xml(self, records: List[DDSRecord], output_path: str) -> str:
        """Export DDS records to TRACES NT-compatible XML."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        root = ET.Element("EUDRDueDiligenceStatements")
        root.set("xmlns", "urn:eu:europa:ec:eudr:dds:1.0")
        root.set("schemaVersion", "1.0")
        root.set("exportedAt", datetime.now(timezone.utc).isoformat())

        for r in records:
            stmt = ET.SubElement(root, "Statement")
            ET.SubElement(stmt, "DDSReference").text = r.dds_reference
            ET.SubElement(stmt, "SubmissionDate").text = r.submission_date
            ET.SubElement(stmt, "RegulationReference").text = "EU 2023/1115"

            op = ET.SubElement(stmt, "Operator")
            ET.SubElement(op, "Name").text = r.operator.name
            ET.SubElement(op, "Address").text = r.operator.address
            ET.SubElement(op, "CountryCode").text = r.operator.country_iso2
            ET.SubElement(op, "EORI").text = r.operator.eori
            ET.SubElement(op, "Email").text = r.operator.email

            com = ET.SubElement(stmt, "Commodity")
            ET.SubElement(com, "HSCode").text = r.commodity.hs_code
            ET.SubElement(com, "Description").text = r.commodity.description
            ET.SubElement(com, "Quantity").text = str(r.commodity.quantity)
            ET.SubElement(com, "Unit").text = r.commodity.unit
            period = ET.SubElement(com, "ProductionPeriod")
            ET.SubElement(period, "Start").text = r.commodity.production_start
            ET.SubElement(period, "End").text = r.commodity.production_end

            geo = ET.SubElement(stmt, "Geolocation")
            ET.SubElement(geo, "FarmId").text = r.farm_id
            ET.SubElement(geo, "CountryOfProduction").text = r.country_of_production
            ET.SubElement(geo, "Latitude").text = str(r.latitude)
            ET.SubElement(geo, "Longitude").text = str(r.longitude)
            ET.SubElement(geo, "AreaHa").text = str(r.area_ha)

            ra = ET.SubElement(stmt, "RiskAssessment")
            ET.SubElement(ra, "AssessmentDate").text = r.assessment_date
            ET.SubElement(ra, "BaselineYear").text = str(r.baseline_year)
            ET.SubElement(ra, "AssessmentYear").text = str(r.assessment_year)
            ET.SubElement(ra, "ModelVersion").text = r.model_version
            ET.SubElement(ra, "Conclusion").text = r.risk_level
            ET.SubElement(ra, "DeforestationPercent").text = str(r.deforestation_pct)
            ET.SubElement(ra, "EvidenceHash").text = r.evidence_hash
            ET.SubElement(ra, "Notes").text = r.notes

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        logger.info("DDS XML written to %s (%d records)", output_path, len(records))
        return output_path

    # ------------------------------------------------------------------
    # PDF export  (human-readable summary)
    # ------------------------------------------------------------------

    def to_pdf(self, records: List[DDSRecord], output_path: str) -> str:
        """Export a human-readable DDS summary PDF. Requires reportlab."""
        if not _REPORTLAB:
            raise ImportError(
                "reportlab is required for PDF export: pip install reportlab"
            )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
        )

        story = []
        title_style = styles["Title"]
        heading_style = styles["Heading2"]
        normal_style = styles["Normal"]

        story.append(Paragraph("EU Deforestation Regulation — Due Diligence Statement", title_style))
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(f"Regulation: EU 2023/1115", normal_style))
        story.append(Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", normal_style))
        story.append(Spacer(1, 0.6 * cm))

        # Operator block
        if records:
            op = records[0].operator
            story.append(Paragraph("Operator", heading_style))
            for line in [
                f"Name: {op.name}",
                f"Address: {op.address}",
                f"Country: {op.country_iso2}",
                f"EORI: {op.eori or '—'}",
                f"Email: {op.email or '—'}",
            ]:
                story.append(Paragraph(line, normal_style))
            story.append(Spacer(1, 0.5 * cm))

        # Summary counts
        counts = {"COMPLIANT": 0, "WARNING": 0, "VIOLATION": 0}
        for r in records:
            counts[r.risk_level] = counts.get(r.risk_level, 0) + 1

        story.append(Paragraph("Assessment Summary", heading_style))
        summary_data = [
            ["Metric", "Count"],
            ["Total farms assessed", str(len(records))],
            ["Compliant", str(counts.get("COMPLIANT", 0))],
            ["Warning", str(counts.get("WARNING", 0))],
            ["Violation", str(counts.get("VIOLATION", 0))],
        ]
        summary_table = Table(summary_data, colWidths=[10 * cm, 4 * cm])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C5F2E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.6 * cm))

        # Per-farm table
        story.append(Paragraph("Farm-Level Results", heading_style))
        table_data = [["Farm ID", "Country", "Deforestation %", "Risk Level", "DDS Reference"]]

        _RISK_COLORS = {
            "COMPLIANT": colors.HexColor("#D4EDDA"),
            "WARNING": colors.HexColor("#FFF3CD"),
            "VIOLATION": colors.HexColor("#F8D7DA"),
        }

        row_colors = []
        for r in records:
            table_data.append([
                r.farm_id,
                r.country_of_production,
                f"{r.deforestation_pct:.1f}%",
                r.risk_level,
                r.dds_reference,
            ])
            row_colors.append(_RISK_COLORS.get(r.risk_level, colors.white))

        farm_table = Table(
            table_data,
            colWidths=[4.5 * cm, 2.5 * cm, 3 * cm, 3 * cm, 4 * cm],
        )
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C5F2E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        for i, bg in enumerate(row_colors, start=1):
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
        farm_table.setStyle(TableStyle(style_cmds))
        story.append(farm_table)
        story.append(Spacer(1, 0.8 * cm))

        # Evidence block
        if records:
            story.append(Paragraph("Evidence & Audit", heading_style))
            story.append(Paragraph(f"Model version: {records[0].model_version}", normal_style))
            story.append(Paragraph(f"Baseline year: {records[0].baseline_year}", normal_style))
            story.append(Paragraph(f"Assessment year: {records[0].assessment_year}", normal_style))
            story.append(Paragraph(f"Evidence hash: {records[0].evidence_hash}", normal_style))

        doc.build(story)
        logger.info("DDS PDF written to %s", output_path)
        return output_path
