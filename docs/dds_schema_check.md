# DDS export against the official schema (E4)

Checked 12 Nov 2026. The README called `dds.json` and `dds.xml` compatible with the EU Information System and TRACES NT.
They are not, so the wording now says "modelled on".

## How the official format was read

- The EU Information System documentation for operators, "EUDR Operator API Reference", documentation version 1.0
  (29 May 2026), describes DDS submission as a SOAP web service, DDS V3:
  https://eudr.webcloud.ec.europa.eu/tracesnt/help/eudr-documentation/operator/index.html and
  https://acceptance.eudr.webcloud.ec.europa.eu/tracesnt/help/eudr-documentation/operator/api/due-diligence-statement-v3.html
- The schema was downloaded from the acceptance environment's published XSDs,
  `https://acceptance.eudr.webcloud.ec.europa.eu/tracesnt/ws/EUDRDueDiligenceStatementServiceV3?xsd=1` to `?xsd=4`
  (6,613, 3,665, 24,804 and 18,021 bytes). They are not copied into this repo.

## What differs

| | official DDS V3 | `src/dds_exporter.py` |
|---|---|---|
| transport | SOAP request (`SubmitDdsRequest`, `AmendDdsRequest`, ...) with WS-Security | files on disk |
| namespace | `http://ec.europa.eu/tracesnt/certificate/eudr/due-diligence-statement/v3` (plus common/v3, body/v3, base/v4) | `urn:eu:europa:ec:eudr:dds:1.0`, not used by the Information System |
| top level | one `statement` per request: `activityType`, `countryOfActivity`, `commodities` (`descriptors`, `hsHeading`, `speciesInfo`, `producers`), `geoLocationConfidential`, ... | `EUDRDueDiligenceStatements` with one `Statement` per farm |
| risk result | no field for a model's conclusion | `RiskAssessment` (`Conclusion`, `DeforestationPercent`, `ModelVersion`, ...) |

The JSON export has no official counterpart: the Information System's documented interface is SOAP/XML.

## What a real export would need

A mapping from one report to one `statement` per consignment (not per farm), the producers' geolocation in the form the
V3 schema asks for, the operator's identifiers, and a schema test against the published XSDs in CI. The model's risk
result would stay outside the statement, in the operator's own due-diligence records. Whether and how plot geolocation
must be given (points or polygons by plot size) is a question for the regulation text and the owner, as for E2 and E5.
