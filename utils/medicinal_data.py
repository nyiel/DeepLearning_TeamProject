# utils/medicinal_data.py

MEDICINAL_DB = {
    'Neem': {
        'english_name': 'Neem Tree',
        'scientific_name': 'Azadirachta indica',
        'benefit': 'Powerful antiseptic and antifungal properties. Used for treating skin diseases like acne and eczema. Supports dental health and boosts immunity.'
    },
    
    'Mango': {
        'english_name': 'Mango',
        'scientific_name': 'Mangifera indica',
        'benefit': 'Leaves are used to manage diabetes and blood pressure. Treats respiratory problems like asthma. Aids in healing burns.'
    },
    'Guava': {
        'english_name': 'Guava',
        'scientific_name': 'Psidium guajava',
        'benefit': 'Relieves toothache and gum inflammation. Helps treat diarrhea and dysentery. Lowers bad cholesterol levels.'
    },
    'Lemon': {
        'english_name': 'Lemon',
        'scientific_name': 'Citrus limon',
        'benefit': 'High in Vitamin C, boosts immunity. Acts as a digestive aid. Has antibacterial properties useful for skin infections.'
    }
}

def get_plant_info(plant_name):
    # Default fallback if plant not found
    return MEDICINAL_DB.get(plant_name, {
        'english_name': plant_name,
        'scientific_name': 'Unknown Species', 
        'benefit': 'Medicinal data not currently available for this species.'
    })