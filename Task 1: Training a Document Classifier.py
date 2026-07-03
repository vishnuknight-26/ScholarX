import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    "Text": [
        # Sports
        "I love football",
        "Cricket is exciting",
        "The FIFA World Cup starts today",
        "India won the cricket match",
        "Kabaddi is a popular sport",
        "The football team scored three goals",
        "The tennis tournament begins tomorrow",
        "The player won a gold medal",
        "The basketball league has started",
        "The stadium was full of fans",
        "The coach praised the team",
        "Virat Kohli scored a century",
        "The match went into extra time",
        "The badminton championship is next week",
        "The athlete broke the national record",

        # Technology
        "Artificial Intelligence is amazing",
        "Python is a programming language",
        "Machine learning improves predictions",
        "Cloud computing is becoming popular",
        "Cybersecurity protects computer systems",
        "The software update fixed many bugs",
        "Java is used for backend development",
        "The new smartphone has advanced features",
        "Data science helps businesses make decisions",
        "Robotics is transforming manufacturing",
        "The website loads very quickly",
        "The database stores customer information",
        "The laptop has a powerful processor",
        "The application uses artificial intelligence",
        "Programming requires logical thinking",

        # Politics
        "The election results are out",
        "The government announced a new policy",
        "The parliament passed a new bill",
        "The Prime Minister addressed the nation",
        "The President met foreign leaders",
        "Citizens voted in the general election",
        "The minister launched a welfare scheme",
        "The political party held a public meeting",
        "The budget was presented in parliament",
        "The opposition criticized the new law",
        "The government increased education funding",
        "The election campaign started today",
        "The new policy focuses on healthcare",
        "The chief minister visited the district",
        "The cabinet approved the proposal"
    ],

    "Category": [
        "Sports","Sports","Sports","Sports","Sports",
        "Sports","Sports","Sports","Sports","Sports",
        "Sports","Sports","Sports","Sports","Sports",

        "Technology","Technology","Technology","Technology","Technology",
        "Technology","Technology","Technology","Technology","Technology",
        "Technology","Technology","Technology","Technology","Technology",

        "Politics","Politics","Politics","Politics","Politics",
        "Politics","Politics","Politics","Politics","Politics",
        "Politics","Politics","Politics","Politics","Politics"
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Text"])

y = df["Category"]

model = MultinomialNB()
model.fit(X, y)

new_text = input("Enter a sentence: ")

new_vector = vectorizer.transform([new_text])

prediction = model.predict(new_vector)

print("Predicted Category:", prediction[0])
