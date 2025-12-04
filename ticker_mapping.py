from typing import Optional

COMPANY_TO_TICKER = {
    "Apple": "AAPL",
    "Apple Inc.": "AAPL",
    "Microsoft": "MSFT",
    "Microsoft Corporation": "MSFT",
    "Alphabet": "GOOGL",
    "Alphabet Inc.": "GOOGL",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Amazon.com Inc.": "AMZN",
    "Meta": "META",
    "Meta Platforms": "META",
    "Meta Platforms Inc.": "META",
    "NVIDIA": "NVDA",
    "NVIDIA Corporation": "NVDA",
    "Tesla": "TSLA",
    "Tesla Inc.": "TSLA",
    "Netflix": "NFLX",
    "Netflix Inc.": "NFLX",
    "Adobe": "ADBE",
    "Adobe Inc.": "ADBE",
    "Intel": "INTC",
    "Intel Corporation": "INTC",
    "Advanced Micro Devices": "AMD",
    "AMD": "AMD",
    "JPMorgan": "JPM",
    "JPMorgan Chase": "JPM",
    "JPMorgan Chase & Co.": "JPM",
    "Bank of America": "BAC",
    "Bank of America Corporation": "BAC",
    "Wells Fargo": "WFC",
    "Wells Fargo & Company": "WFC",
    "Goldman Sachs": "GS",
    "The Goldman Sachs Group": "GS",
    "Morgan Stanley": "MS",
    "Citigroup": "C",
    "Citigroup Inc.": "C",
    "Visa": "V",
    "Visa Inc.": "V",
    "Mastercard": "MA",
    "Mastercard Incorporated": "MA",
    "Walmart": "WMT",
    "Walmart Inc.": "WMT",
    "Costco": "COST",
    "Costco Wholesale": "COST",
    "McDonald's": "MCD",
    "McDonald’s": "MCD",
    "Coca-Cola": "KO",
    "The Coca-Cola Company": "KO",
    "PepsiCo": "PEP",
    "PepsiCo Inc.": "PEP",
    "Nike": "NKE",
    "Nike Inc.": "NKE",
    "Starbucks": "SBUX",
    "Exxon Mobil": "XOM",
    "Exxon Mobil Corporation": "XOM",
    "Chevron": "CVX",
    "Chevron Corporation": "CVX",
    "Boeing": "BA",
    "The Boeing Company": "BA",
    "Caterpillar": "CAT",
    "Caterpillar Inc.": "CAT",
    "Johnson & Johnson": "JNJ",
    "Pfizer": "PFE",
    "Pfizer Inc.": "PFE",
    "Eli Lilly": "LLY",
    "Eli Lilly and Company": "LLY",
    "Merck": "MRK",
    "Merck & Co.": "MRK",
    "Berkshire Hathaway": "BRK.B",
    "Berkshire Hathaway Inc.": "BRK.B",
    "Disney": "DIS",
    "The Walt Disney Company": "DIS",
    "Salesforce": "CRM",
    "Salesforce Inc.": "CRM",
    "PayPal": "PYPL",
    "PayPal Holdings": "PYPL",
}


def _normalize_name(name: str) -> str:
    if not name:
        return ""
    n = name.replace(",", " ").strip()
    for suffix in ["Inc.", "Inc", "Corp.", "Corp", "Corporation", "PLC", "Ltd.", "Ltd"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def map_company_to_ticker(company_name: str) -> Optional[str]:
    if not company_name:
        return None

    if company_name in COMPANY_TO_TICKER:
        return COMPANY_TO_TICKER[company_name]

    normalized = _normalize_name(company_name)
    for key, ticker in COMPANY_TO_TICKER.items():
        if _normalize_name(key).lower() == normalized.lower():
            return ticker

    return None
