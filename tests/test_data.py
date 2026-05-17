"""
Test dataset: queries with expected relevant subjects.
Used to evaluate retrieval quality.
"""

TEST_QUERIES = [
    {
        "query": "Messi vs Ronaldo career achievements",
        "domain": "football",
        "expected_subjects": ["Lionel Messi", "Cristiano Ronaldo"],
        "description": "Direct comparison of two top players"
    },
    {
        "query": "best football strikers in the world",
        "domain": "football",
        "expected_subjects": ["Cristiano Ronaldo", "Erling Haaland", "Kylian Mbappé"],
        "description": "Generic query about strikers"
    },
    {
        "query": "Inception vs The Dark Knight plot comparison",
        "domain": "movies",
        "expected_subjects": ["Inception", "The Dark Knight"],
        "description": "Direct movie comparison"
    },
    {
        "query": "most influential musicians in history",
        "domain": "music",
        "expected_subjects": ["The Beatles", "Michael Jackson", "Bob Dylan"],
        "description": "Generic music query"
    },
    {
        "query": "Messi career stats and achievements",
        "domain": "football",
        "expected_subjects": ["Lionel Messi"],
        "description": "Single subject query"
    },
]
