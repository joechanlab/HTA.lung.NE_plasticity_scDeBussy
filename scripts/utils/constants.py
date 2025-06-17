sample_groups = {
    "De novo SCLC and ADC": [
        "RU1311", "RU1124", "RU1108", "RU1065",
        "RU871", "RU1322", "RU1144", "RU1066",
        "RU1195", "RU779", "RU1152", "RU426",
        "RU1231", "RU222", "RU1215", "RU1145",
        "RU1360", "RU1287", # 18
    ],
    "ADC → SCLC": [
        "RU1068", "RU1676", "RU263", "RU942",
        "RU325", "RU1304", "RU1444", "RU151",
        "RU1181", "RU1042", "RU581", "RU1303",
        "RU1305", "RU1250", "RU831", "RU1646",
        "RU226", "RU773", "RU1083", "RU1518",
        "RU1414", "RU1405", "RU1293" # 23
    ]
}
sample_to_group = {
    sample_id: group
    for group, samples in sample_groups.items()
    for sample_id in samples
}

tumor_color_map = {
    "NSCLC": "gold",      # Yellow
    "SCLC-A": "tab:red",      # Red
    "SCLC-N": "tab:cyan",      # Cyan
    "SCLC-P": "tab:blue",      # Blue
    "NonNE\nSCLC": "tab:purple",  # Purple
    "NonNE SCLC": "tab:purple",  # Purple
    "pDC": "#004d00",
    "PMN": "#339933",
    "Mφ/Mono": "#66cc66",
    "Mφ/Mono\nCD14": "#99cc99",
    "Mφ/Mono\nCD11c": "#336633",
    "Mast cell": "#003300"
}