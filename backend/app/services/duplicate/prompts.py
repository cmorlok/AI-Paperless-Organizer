# Placeholder: {content}
INVOICE_EXTRACTION_PROMPT = (
    "Extrahiere aus dem folgenden Dokumenttext die Rechnungsnummer und den Gesamtbetrag.\n"
    "Antworte NUR mit einem JSON-Objekt im Format:\n"
    '{"invoice_number": "...", "amount": "..."}\n'
    "Wenn du keine Rechnungsnummer findest, sette den Wert auf einen leeren String.\n"
    "Wenn du keinen Betrag findest, sette den Wert auf einen leeren String.\n\n"
    "Dokumenttext:\n{content}"
)
