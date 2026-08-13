import os

# Create assignments folder
os.makedirs("assignments", exist_ok=True)

assignments = {

    "assignment_A.txt": """
Artificial intelligence is a branch of computer science that focuses on
creating systems capable of performing tasks that normally require human
intelligence. Machine learning is an important part of artificial intelligence.
It allows computers to learn patterns from data and make predictions without
being explicitly programmed. Artificial intelligence is widely used in
healthcare, banking, education, transportation and customer service.
""",

    "assignment_B.txt": """
Artificial intelligence is a branch of computer science that focuses on
creating systems capable of performing tasks that normally require human
intelligence. Machine learning is an important part of artificial intelligence.
It allows computers to learn patterns from data and make predictions without
being explicitly programmed. Artificial intelligence is widely used in
healthcare, banking, education, transportation and customer service.
Machine learning has become an important technology in modern applications.
""",

    "assignment_C.txt": """
Natural language processing is a field of artificial intelligence that deals
with the interaction between computers and human language. NLP techniques
allow computers to understand, process and generate human language. Common
applications include chatbots, machine translation, sentiment analysis,
speech recognition and text summarization. NLP is widely used in modern
communication systems.
""",

    "assignment_D.txt": """
Artificial intelligence is widely used in modern technology. Machine learning
allows computers to identify patterns in data and make predictions. These
technologies are used in healthcare, banking, transportation and education.
Artificial intelligence has become an important part of many computer systems.
""",

    "assignment_E.txt": """
Cloud computing provides computing resources through the internet. Users can
access storage, servers, databases and software without maintaining physical
hardware. Cloud services can reduce infrastructure costs and provide flexible
resources. Organizations use cloud computing for data storage, application
development, backup and business operations.
""",

    "assignment_F.txt": """
Natural language processing is a field of artificial intelligence that deals
with human language. NLP enables computers to understand and process text
and speech. Applications of NLP include chatbots, machine translation,
sentiment analysis, speech recognition and text summarization. Natural
language processing is widely used in communication and information systems.
""",

    "assignment_G.txt": """
Cybersecurity is the practice of protecting computers, networks and data from
unauthorized access and attacks. Strong passwords, encryption, firewalls and
security monitoring can help organizations protect sensitive information.
Cybersecurity is important because organizations store large amounts of
personal and financial data on computer systems.
""",

    "assignment_H.txt": """
Renewable energy comes from natural resources that can be replenished over
time. Solar energy, wind energy, hydropower and biomass are common examples.
Renewable energy can reduce dependence on fossil fuels and help reduce
greenhouse gas emissions. Governments and organizations are investing in
renewable energy technologies for a sustainable future.
"""
}

# Write files
for filename, content in assignments.items():

    path = os.path.join("assignments", filename)

    with open(path, "w", encoding="utf-8") as file:
        file.write(content.strip())

    print("Created:", path)

print("\nDataset created successfully!")
print("Total documents:", len(assignments))
