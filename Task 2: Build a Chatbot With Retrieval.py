import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    "Question": [
        "What is Python?",
        "Why should I learn Java?",
        "What is SQL?",
        "What is DBMS?",
        "What is MySQL?",
        "What is Machine Learning?",
        "What is Artificial Intelligence?",
        "What is Data Science?",
        "What is HTML?",
        "What is CSS?",
        "What is JavaScript?",
        "What is Git?",
        "What is GitHub?",
        "What is an API?",
        "What is OOP?",
        "What is a Class?",
        "What is an Object?",
        "What is a Function?",
        "What is a Variable?",
        "What is a Loop?",
        "What is an Array?",
        "What is Exception Handling?",
        "What is Linux?",
        "What is Cloud Computing?",
        "What is Cybersecurity?",
        "What is a Computer Network?",
        "What is a Compiler?",
        "What is an Interpreter?",
        "Difference between Java and Python",
        "What is a Resume?"
    ],

    "Answer": [
        "Python is a beginner-friendly programming language used for web development, AI, automation, and data science.",
        "Java is widely used for backend development, Android applications, and enterprise software.",
        "SQL is a language used to create, retrieve, update, and manage data in databases.",
        "DBMS stands for Database Management System. It helps store and organize data efficiently.",
        "MySQL is a popular relational database management system used in many web applications.",
        "Machine Learning is a branch of AI that enables computers to learn from data and make predictions.",
        "Artificial Intelligence enables machines to perform tasks that normally require human intelligence.",
        "Data Science is the process of analyzing data to discover useful information and insights.",
        "HTML is the standard markup language used to create web pages.",
        "CSS is used to style and design web pages.",
        "JavaScript is used to make websites interactive and dynamic.",
        "Git is a version control system used to track changes in source code.",
        "GitHub is an online platform used to store, manage, and share Git repositories.",
        "An API allows two software applications to communicate with each other.",
        "Object-Oriented Programming (OOP) is a programming concept based on classes and objects.",
        "A class is a blueprint used to create objects in object-oriented programming.",
        "An object is an instance of a class containing data and methods.",
        "A function is a reusable block of code that performs a specific task.",
        "A variable is a named memory location used to store data.",
        "A loop executes a block of code repeatedly until a condition is met.",
        "An array stores multiple values of the same data type in a single variable.",
        "Exception handling is used to detect and handle runtime errors in a program.",
        "Linux is an open-source operating system commonly used in servers and software development.",
        "Cloud Computing provides storage, servers, and software over the internet.",
        "Cybersecurity protects computers, networks, and data from cyber attacks.",
        "A computer network connects multiple computers to share data and resources.",
        "A compiler converts the entire source code into machine code before execution.",
        "An interpreter translates and executes source code line by line.",
        "Java is a compiled, object-oriented language, while Python is an interpreted language with simpler syntax.",
        "A resume is a document that highlights your education, skills, projects, and achievements."
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df["Question"])

print("=" * 60)
print("               COLLEGE STUDENT AI CHATBOT")
print("=" * 60)
print("Ask any programming or computer science question.")
print("Type 'exit' to close the chatbot.\n")

while True:

    user_question = input("You: ")

    if user_question.lower() == "exit":
        print("\nChatbot: Thank you! Happy Learning!")
        break

    user_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(user_vector, question_vectors)

    best_match = similarity.argmax()

    score = similarity[0][best_match]

    if score > 0.30:
        print("\nChatbot:", df["Answer"][best_match])
    else:
        print("\nChatbot: Sorry, I don't know the answer to that question yet.")
