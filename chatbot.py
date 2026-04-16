responses = {
    ("hello", "hi", "hey"): "Hi there!",
    ("how are you", "how are u"): "I am fine. Thanks for asking!",
    ("what is your name", "your name", "name"): "I am a rule-based AI chatbot.",
    ("help", "assist me"): "You can greet me, ask my name, or type exit to quit.",
    ("bye", "goodbye"): "Goodbye! It was nice talking to you.",
    ("thanks", "thank you"): "You're welcome!"
}

print("====================================")
print("   Welcome to Rule-Based Chatbot")
print("====================================")
print("Chatbot: Hello! I am your AI chatbot.")
print("Chatbot: Type 'exit' anytime to quit.\n")

while True:
    raw_input_text = input("You: ")
    user_input = raw_input_text.lower().strip()

    if user_input == "exit":
        print("Chatbot: Goodbye! Have a great day.")
        break

    found = False

    for keys, value in responses.items():
        if user_input in keys:
            print("Chatbot:", value)
            found = True
            break

    if not found:
        print("Chatbot: Sorry, I do not understand that.")