OCR_SYSTEM_MESSAGE = (
    "You are a precise OCR module. Output ONLY the verbatim transcribed text from the image – nothing else. "
    "No summaries, no descriptions, no commentary, no 'Let me...', no 'Here is...'. "
    "Every number, every EUR amount, every date, every code must appear EXACTLY as printed. "
    "For tables: every row, every column, every cell value. "
    "Missing a single number is a critical OCR failure. Raw verbatim transcription only."
)

OCR_ANTI_TABLE_PROMPT = (
    "Transcribe ALL text in this image completely from top to bottom. "
    "Do NOT use table formatting, pipes |, or dashes ---. "
    "Write each piece of information on its own line, using colons for labels. "
    "Include every single line of text: headers, items, prices, totals, footer, company details, IBAN."
)

OCR_DEEPSEEK_PROMPT = "OCR this document."
OCR_GLM_PROMPT = "Text Recognition:"
OCR_GEMMA3_PROMPT = (
    "Just transcribe the text in this image. Preserve the formatting and layout. "
    "Be thorough, continue until the bottom of the page. "
    "Use markdown format but without a code block."
)
OCR_DEFAULT_PARTS = [
    "Transcribe ALL text in this image EXACTLY as it appears – high quality OCR.",
    "CRITICAL: Do NOT summarize, skip, or abbreviate any content. Continue until the very bottom of the page.",
    "CRITICAL: Every single number, amount, percentage, account number, and code MUST be transcribed exactly.",
    "For tables: transcribe each row completely, including all columns and values.",
    "For checkboxes/tick boxes: write [ ] for unchecked and [X] for checked, followed by the label text.",
    "For form fields: write the label followed by the filled-in value or a blank line if empty.",
    "For structured forms (tax notices, invoices, bank statements): preserve every field label and its value.",
    "Use markdown format without code blocks. Preserve the original layout as closely as possible.",
]
OCR_DEFAULT_GERMAN_HINT = (
    "The document is in German. "
    "Pay special attention to: names, dates (DD.MM.YYYY), IBANs, BIC codes, "
    "tax IDs (Steuernummer), amounts in EUR, account numbers, reference numbers, "
    "and addresses. Transcribe every value exactly as printed – no rounding, no omitting."
)


# Placeholders: {{ DOCUMENT_TITLE }}, {{ VERSION_COUNT }}, {{ MODELS_TEXT }}
OCR_EVALUATION_PROMPT = """Du bist ein erfahrener OCR-Qualitätsprüfer und Dokumentenanalyst. Du bewertest OCR-Ergebnisse für ein deutsches Dokumentenmanagementsystem (Paperless-ngx).

DOKUMENT: "{{ DOCUMENT_TITLE }}"
ANZAHL VERSIONEN: {{ VERSION_COUNT }}

Folgende OCR-Versionen desselben Dokuments wurden von verschiedenen lokalen Vision-Modellen (Ollama) erstellt. Vergleiche sie gründlich.

{{ MODELS_TEXT }}

BEWERTUNGSANLEITUNG:
Du musst jede Version sorgfältig auf folgende Kriterien prüfen. Vergleiche die Versionen untereinander -- wenn mehrere Versionen den gleichen Wert haben, ist er wahrscheinlich korrekt. Abweichungen deuten auf Fehler hin.

KRITISCHE FELDER (Fehler hier = sofortiger Punktabzug):
- Namen (Vor-/Nachname): Auch ein einziger falscher Buchstabe ist ein Fehler
- Datumsangaben: Falsches Jahr/Monat = KO-Kriterium (schlimmer als Tippfehler!)
- IBAN/Kontonummern: Ziffern müssen exakt stimmen, Leerzeichen-Gruppierung egal
- Geldbeträge: Müssen exakt stimmen

WICHTIGE FELDER:
- Adressen, Zählernummern, Referenznummern
- Checkbox-Zustände (angekreuzt vs. leer)
- Formularlogik (Felder richtig zugeordnet?)

ALLGEMEINE QUALITÄT:
- Vollständigkeit (fehlen Textblöcke/Absätze?)
- Halluzinationen (hat das Modell Text erfunden der nicht im Original steht?)
- Wiederholungen (Textblöcke die sich wiederholen)
- Formatierung und Lesbarkeit

PRAXISTAUGLICHKEIT:
- Kann der Text automatisiert weiterverarbeitet werden?
- Wie viel manuelle Nacharbeit wäre nötig?

Anworte NUR mit validem JSON (kein Text davor/danach, keine Markdown-Codeblöcke):
{{
  "ranking": [
    {{
      "rank": 1,
      "model": "<modellname>",
      "overall_score": <0-100>,
      "category_scores": {{
        "names_persons": <0-10>,
        "dates_periods": <0-10>,
        "iban_banking": <0-10>,
        "amounts_numbers": <0-10>,
        "addresses": <0-10>,
        "form_logic": <0-10>,
        "completeness": <0-10>,
        "formatting": <0-10>,
        "no_hallucinations": <0-10>,
        "automatizability": <0-10>
      }},
      "speed_seconds": <dauer>,
      "strengths": ["Stärke 1", "Stärke 2"],
      "weaknesses": ["Schwäche 1"],
      "specific_errors": [
        {{"field": "Name", "expected": "korrekt", "got": "was das Modell geschrieben hat", "severity": "critical"}},
        {{"field": "IBAN", "expected": "DE12 3456...", "got": "DE12 3546...", "severity": "high"}}
      ],
      "verdict": "<1-2 Sätze Praxisurteil auf Deutsch>"
    }}
  ],
  "best_quality": "<modellname mit bester Qualität>",
  "best_speed": "<schnellstes Modell>",
  "best_value": "<bestes Preis-Leistungs-Verhältnis (Qualität vs. Geschwindigkeit)>",
  "recommendation": "<3-4 Sätze Empfehlung auf Deutsch: welches Modell für Produktion, welches Backup, welches nicht verwenden>",
  "critical_finding": "<wichtigste Erkenntnis, z.B. 'Datumsfehler bei Modell X sind ein KO-Kriterium'>",
  "cross_comparison": {{
    "agreement": ["Felder wo alle Versionen übereinstimmen"],
    "disagreement": ["Felder wo die Versionen sich widersprechen -- hier liegt wahrscheinlich mindestens ein Fehler"]
  }}
}}

WICHTIG:
- Severity-Stufen: "critical" (Daten, Namen, IBAN falsch), "high" (wichtige Felder), "medium" (Formatierung), "low" (kosmetisch)
- Score 0-100: unter 50 = nicht verwendbar, 50-70 = bedingt brauchbar, 70-85 = gut, 85+ = sehr gut
- Sei STRENG aber FAIR. Ein falsches Datum ist schlimmer als 5 Tippfehler.
- Wenn du nicht sicher bist ob ein Wert richtig ist, vergleiche die Versionen untereinander.
"""

PROMPTS = {
    "ocr_evaluation": OCR_EVALUATION_PROMPT,
}
