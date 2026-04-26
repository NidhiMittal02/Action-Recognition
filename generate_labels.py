import json

def generate_label_map():
    labels = [
        "drink water", "eat meal", "brush teeth", "brush hair", "drop",
        "pick up", "throw", "sit down", "stand up", "clapping",
        "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
        "wear shoe", "take off shoe", "wear glasses", "take off glasses",
        "put on hat", "take off hat", "cheer up", "hand waving",
        "kicking something", "reach into pocket", "hopping", "jump up",
        "make phone call", "play with phone", "type on keyboard",
        "point to something", "taking selfie", "check time", "rub hands",
        "nod head", "shake head", "wipe face", "salute",
        "put palms together", "cross hands", "sneeze or cough",
        "staggering", "falling down", "headache", "chest pain",
        "back pain", "neck pain", "nausea", "fan self",
        "punch or slap", "kicking", "pushing", "pat on back",
        "point finger", "hugging", "giving object", "touch pocket",
        "shaking hands", "walking", "standing"
    ]

    # 🔥 IMPORTANT FIX: keys must be strings
    label_map = {str(i): label for i, label in enumerate(labels)}

    with open("label_map.json", "w") as f:
        json.dump(label_map, f, indent=4)

    print("✅ label_map.json created successfully!")

if __name__ == "__main__":
    generate_label_map()