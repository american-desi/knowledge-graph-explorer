"""
Built-in text corpus: paragraphs about tech companies, historical events,
and scientific discoveries. 25+ substantial paragraphs covering diverse topics
with rich entity/relationship content for knowledge graph extraction.
"""

CORPUS = [
    # ── Tech Companies & Founders ─────────────────────────────────────────

    (
        "Elon Musk co-founded Tesla in San Francisco in 2003 and serves as its CEO. "
        "Tesla is headquartered in Austin, Texas and develops electric vehicles and "
        "autonomous driving technology. Musk also founded SpaceX in Hawthorne, "
        "California, which develops reusable rockets and spacecraft. SpaceX competes "
        "with Boeing and Lockheed Martin in the space launch industry."
    ),

    (
        "Larry Page and Sergey Brin co-founded Google in 1998 while they were PhD "
        "students at Stanford in California. Google, now a subsidiary of Alphabet, "
        "is headquartered in Mountain View and developed the PageRank algorithm that "
        "revolutionized web search. Sundar Pichai serves as CEO of both Google and "
        "Alphabet. Google also developed Android, Chrome, and TensorFlow."
    ),

    (
        "Mark Zuckerberg founded Facebook in 2004 while studying at Harvard. "
        "The company, now called Meta, is headquartered in Menlo Park, California. "
        "Meta acquired Instagram in 2012 and WhatsApp in 2014. Zuckerberg leads "
        "the company's push into virtual reality and the metaverse. Meta competes "
        "with Google and Apple in the technology sector."
    ),

    (
        "Steve Jobs and Steve Wozniak co-founded Apple in 1976 in a garage in "
        "Cupertino, California. Apple developed the Macintosh, iPhone, and iPad, "
        "transforming personal computing and mobile technology. Tim Cook became "
        "CEO of Apple after Jobs passed away in 2011. Apple is headquartered in "
        "Cupertino and competes with Samsung, Google, and Microsoft."
    ),

    (
        "Bill Gates and Paul Allen co-founded Microsoft in 1975 in Redmond, "
        "Washington. Microsoft developed Windows, the most widely used operating "
        "system in the world. Satya Nadella became CEO of Microsoft in 2014 and "
        "led the company's transformation toward cloud computing with Azure. "
        "Microsoft acquired LinkedIn in 2016 and GitHub in 2018."
    ),

    (
        "Jeff Bezos founded Amazon in 1994 in Seattle, Washington. Amazon started "
        "as an online bookstore but grew into the world's largest e-commerce company. "
        "Andy Jassy became CEO of Amazon in 2021. Amazon developed AWS, the leading "
        "cloud computing platform, and acquired Whole Foods in 2017. Amazon competes "
        "with Microsoft and Google in cloud computing."
    ),

    # ── AI & Machine Learning ─────────────────────────────────────────────

    (
        "Sam Altman leads OpenAI, which is based in San Francisco. OpenAI developed "
        "ChatGPT and GPT-4, large language models that have transformed artificial "
        "intelligence. Microsoft invested in OpenAI and partnered with the company "
        "to integrate AI into its products. OpenAI competes with Google DeepMind "
        "and Anthropic in the AI industry."
    ),

    (
        "Demis Hassabis co-founded DeepMind in London in 2010. Google acquired "
        "DeepMind in 2014 for over 500 million dollars. DeepMind developed "
        "AlphaGo, which defeated world champions in the game of Go using deep "
        "learning and reinforcement learning. Hassabis leads research into "
        "artificial general intelligence at DeepMind."
    ),

    (
        "Geoffrey Hinton, Yann LeCun, and Yoshua Bengio are known as the "
        "pioneers of deep learning. Hinton worked at Google and the University "
        "of Toronto, LeCun leads AI research at Meta, and Bengio is a professor "
        "in Montreal, Canada. Together they developed neural network techniques "
        "including backpropagation that form the foundation of modern machine learning."
    ),

    (
        "Jensen Huang co-founded NVIDIA in 1993 in Santa Clara, California. "
        "NVIDIA developed the GPU, which became essential for deep learning "
        "and artificial intelligence. NVIDIA is headquartered in Santa Clara "
        "and competes with AMD and Intel in the semiconductor industry. Lisa Su "
        "serves as CEO of AMD, which is based in Santa Clara."
    ),

    # ── Internet & Social Media ───────────────────────────────────────────

    (
        "Tim Berners-Lee invented the World Wide Web in 1989 while working at "
        "CERN in Geneva, Switzerland. He developed HTML, HTTP, and the first "
        "web browser, laying the foundation for the modern Internet. Berners-Lee "
        "later founded the World Wide Web Consortium to develop web standards. "
        "Vint Cerf, working at DARPA, co-developed TCP/IP, the protocol that "
        "underpins the Internet."
    ),

    (
        "Jack Dorsey co-founded Twitter in San Francisco in 2006. Twitter "
        "became a major social media platform before Elon Musk acquired the "
        "company in 2022. Reid Hoffman co-founded LinkedIn in Mountain View, "
        "which Microsoft acquired in 2016. Kevin Systrom co-founded Instagram "
        "in San Francisco, and Meta acquired Instagram in 2012."
    ),

    (
        "Daniel Ek co-founded Spotify in Stockholm, Sweden in 2006. Spotify "
        "developed a music streaming platform that competes with Apple Music "
        "and Amazon Music. Reed Hastings co-founded Netflix in Los Angeles, "
        "which pioneered video streaming and competes with Disney and Amazon "
        "in the entertainment industry."
    ),

    # ── Computing Pioneers ────────────────────────────────────────────────

    (
        "Alan Turing, born in London in 1912, is considered the father of "
        "computer science. He developed the concept of the Turing machine at "
        "Cambridge and worked on breaking the Enigma code during World War II. "
        "Grace Hopper developed one of the first compilers and contributed to "
        "the development of COBOL while working at Harvard and later for the "
        "United States Navy."
    ),

    (
        "John von Neumann, born in Budapest, developed the von Neumann "
        "architecture that forms the basis of modern computers. He worked at "
        "Princeton and contributed to the Manhattan Project. Claude Shannon, "
        "working at Bell Labs in New Jersey, founded information theory and "
        "laid the mathematical foundations for digital computing."
    ),

    (
        "Linus Torvalds created Linux in 1991 while a student at the University "
        "of Helsinki in Finland. Linux became the foundation for Android, cloud "
        "servers, and supercomputers worldwide. Guido van Rossum created Python "
        "in the Netherlands in 1991, and Python became the most popular "
        "programming language for machine learning and data science."
    ),

    # ── Semiconductors & Hardware ─────────────────────────────────────────

    (
        "Gordon Moore and Robert Noyce co-founded Intel in 1968 in Santa Clara, "
        "California. Moore formulated Moore's Law, predicting that the number of "
        "transistors on microprocessors would double every two years. Intel "
        "developed the first commercial microprocessor and dominated the CPU "
        "market for decades. Pat Gelsinger serves as CEO of Intel."
    ),

    (
        "Samsung, headquartered in Seoul, South Korea, is the world's largest "
        "semiconductor manufacturer. Samsung competes with Apple in smartphones "
        "and with Intel in chip manufacturing. Taiwan's TSMC manufactures chips "
        "for Apple, NVIDIA, and AMD. The semiconductor industry is concentrated "
        "in Silicon Valley, Taiwan, and South Korea."
    ),

    # ── Science & Discovery ───────────────────────────────────────────────

    (
        "Albert Einstein, born in Germany in 1879, developed the theory of "
        "relativity while working in Switzerland. He later moved to Princeton "
        "in the United States. Marie Curie, born in Poland, conducted pioneering "
        "research on radioactivity in Paris, France. She was the first woman to "
        "win a Nobel Prize and remains one of the most influential scientists."
    ),

    (
        "CRISPR-Cas9 gene editing technology was developed by researchers at "
        "Berkeley in California. This technology revolutionized biotechnology "
        "and has applications in medicine, agriculture, and genetics. Pfizer "
        "and Moderna used mRNA technology to develop COVID-19 vaccines, "
        "demonstrating the power of biotechnology in public health."
    ),

    # ── Enterprise Software ───────────────────────────────────────────────

    (
        "Larry Ellison founded Oracle in 1977 in Redmond, California. Oracle "
        "developed relational database technology that became essential for "
        "enterprise computing. Marc Benioff founded Salesforce in San Francisco "
        "in 1999, pioneering cloud-based customer relationship management. "
        "Salesforce competes with Oracle and Microsoft in enterprise software."
    ),

    (
        "Stewart Butterfield co-founded Slack in San Francisco, which "
        "Salesforce acquired in 2021. Slack developed a team collaboration "
        "platform that competes with Microsoft Teams. GitHub, founded in "
        "San Francisco, was acquired by Microsoft in 2018. GitHub developed "
        "a platform for collaborative software development using Git."
    ),

    # ── Autonomous Vehicles & Transportation ──────────────────────────────

    (
        "Tesla, led by Elon Musk, is developing self-driving technology in "
        "Austin, Texas. Google developed Waymo, an autonomous vehicle subsidiary "
        "based in Mountain View, California. Uber, co-founded by Travis Kalanick "
        "in San Francisco, also invested in autonomous vehicles. The self-driving "
        "industry involves artificial intelligence, computer vision, and LiDAR "
        "technology."
    ),

    # ── Fintech & E-Commerce ──────────────────────────────────────────────

    (
        "Peter Thiel and Max Levchin co-founded PayPal in San Jose, California "
        "in 1998. Elon Musk's company X.com merged with PayPal before eBay "
        "acquired it. PayPal competes with Stripe, which was founded in "
        "San Francisco and developed payment processing technology. Jack Dorsey "
        "also founded Square in San Francisco for mobile payments."
    ),

    (
        "Jack Ma founded Alibaba in 1999 in China, building the largest "
        "e-commerce platform in Asia. Alibaba competes with Amazon in global "
        "e-commerce. Pony Ma co-founded Tencent in Shenzhen, China, which "
        "developed WeChat and became one of the largest technology companies "
        "in Asia. Tencent and Alibaba compete in cloud computing and digital "
        "payments in China."
    ),

    # ── Quantum Computing & Future Tech ───────────────────────────────────

    (
        "Google, IBM, and Microsoft are investing heavily in quantum computing. "
        "Google claimed quantum supremacy in 2019 with its Sycamore processor "
        "developed in Santa Barbara, California. IBM is developing quantum "
        "computers at its research lab in New York. Microsoft is pursuing "
        "topological quantum computing at its research center in Redmond, "
        "Washington."
    ),
]


def get_all_texts() -> list:
    """Return the complete corpus as a list of text paragraphs."""
    return list(CORPUS)


def get_combined_text() -> str:
    """Return the entire corpus as a single text string."""
    return "\n\n".join(CORPUS)


def get_texts_by_topic(topic: str) -> list:
    """Return texts that mention a specific topic keyword."""
    topic_lower = topic.lower()
    return [text for text in CORPUS if topic_lower in text.lower()]


def get_corpus_stats() -> dict:
    """Return statistics about the corpus."""
    total_chars = sum(len(t) for t in CORPUS)
    total_words = sum(len(t.split()) for t in CORPUS)
    return {
        "num_paragraphs": len(CORPUS),
        "total_characters": total_chars,
        "total_words": total_words,
        "avg_words_per_paragraph": total_words / len(CORPUS),
    }
