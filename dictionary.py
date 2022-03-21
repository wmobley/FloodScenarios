def Residential_Code(x):
    try:
        return {
            'A1': 'RES1',
            'C1': 'RES1',
            'C2': 'RES1',
            'C3': 'RES1',
            'CC': 'RES1',
            'CS': 'RES1',
            'D1': 'RES1',
            'D2': 'RES1',
            'D3': 'RES1',
            'D4': 'RES1',
            'D9': 'RES1',
            'DC': 'RES1',
            'E1': 'RES1',
            'E2': 'RES1',
            'E7': 'RES1',
            'E9': 'RES1',
            'F1': 'RES1',
            'F2': 'RES1',
            'F3': 'RES1',
            'F5': 'RES1',
            'F9': 'RES1',
            'FA': 'RES1',
            'FC': 'RES1',
            'FD': 'RES1',
            'FG': 'RES1',
            'FH': 'RES1',
            'FJ': 'RES1',
            'FM': 'RES1',
            'FN': 'RES1',
            'FO': 'RES1',
            'FR': 'RES1',
            'FS': 'RES1',
            'FT': 'RES1',
            'FY': 'RES1',
            'I1': 'RES1',
            'J2': 'RES1',
            'J3': 'RES1',
            'J4': 'RES1',
            'J5': 'RES1',
            'J6': 'RES1',
            'J7': 'RES1',
            'O1': 'RES1',
            'O2': 'RES1',
            'O2': 'RES1',
            'A2': 'RES2',
            'A3': 'RES2',
            'M3': 'RES2',
            'M4': 'RES2',
            'A4': 'RES3',
            'A5': 'RES3',
            'A7': 'RES3',
            'A9': 'RES3',
            'AC': 'RES3',
            'AG': 'RES3',
            'AJ': 'RES3',
            'AM': 'RES3',
            'AN': 'RES3',
            'AO': 'RES3',
            'AR': 'RES3',
            'AS': 'RES3',
            'AT': 'RES3',
            'AY': 'RES3',
            'Z0': 'RES3',
            'Z1': 'RES3',
            'Z2': 'RES3',
            'Z3': 'RES3',
            'Z4': 'RES3',
            'Z5': 'RES3',
            'B1': 'RES3',
            'B2': 'RES3',
            'B3': 'RES3',
            'B4': 'RES3',
            'B9': 'RES3',
            'BC': 'RES3',
            'BG': 'RES3',
            'BH': 'RES3',
            'BJ': 'RES3',
            'BO': 'RES3',
            'BR': 'RES3',
            'BS': 'RES3',
            'X1': 'RES5',
            'X2': 'RES5',
            'X3': 'RES5',
            'XD': 'RES5',
            'XE': 'RES5',
            'XF': 'RES5',
            'XG': 'RES5',
            'XJ': 'RES5',
            'XL': 'RES5',
            'XU': 'RES5',
            'RES1': 'RES1',
            'RES2': 'RES2',
            'RES3A': 'RES3',
            'RES3B': 'RES3',
            'RES3C': 'RES3',
            'RES3D': 'RES3',
            'RES3E': 'RES3',
            'RES3F': 'RES3',
            'RES3A - F': 'RES3',
            'RES4': 'RES4',
            'RES5': 'RES5',
            'RES6': 'RES6',
        }[x.all()]
    except KeyError:
        return "RES1"


def Foundations(x):
    Pile = 45

    Slab = 65

    try:

        return {

            'PB': Pile,

            'WPR': Pile,

            'CB': Slab,

            'CS': Slab,

            'WPL': Pile,

            'RIB': Slab,

            '9': Slab,

            'S': Slab,

            '4': Slab,

            'CPL': Pile,

            'PT': Pile,

            'Partial Basement': Pile,

            'Crawl Space': Pile,

            'Slab': Slab,

            'Full Basement': Pile

        }[x]

    except:
        return Slab


def exterior(x):
    '''
        Function for changing parcel values to exterior type
        :param x: Pandas dataframe row from Exterior F
        :return: String of exterior type
        '''

    if x == "Aluminum / Vinyl": x = "F"
    if x == "SS": x = "F"
    #     x=x[0]
    try:
        return {
            "A": "Masonry",
            "B": "Masonry",
            "C": "Frame",
            "F": "Frame",
            "G": "Masonry",
            "R": "Masonry",
            "S": "Masonry",
            "T": "Masonry",
            "W": "Frame",
            "MASON": "Masonry",
            "STEEL": "Masonry",
            "CONCR": "Masonry",
            "WOOD": "Frame",
        }[x]
    except:
        return "Masonry"


def exterior_addValue(x):
    if exterior(x) == "Masonry":
        return 5
    else:
        return 0


def economic_life(Quality, Imp_Type):
    if Quality == 'Poor': Quality = "Fair"
    if Quality == 'Very Low': Quality = "Low"
    if Imp_Type == 1001:
        try:
            return {
                "Low": 45,
                "Fair": 50,
                "Average": 55,
                "Good": 55,
                "Superior": 60,
                "Excellent": 60
            }[Quality]
        except:
            return 45
    else:
        try:
            return {
                "Low": 45,
                "Fair": 45,
                "Average": 50,
                "Good": 50,
                "Superior": 55,
                "Excellent": 55
            }[Quality]
        except:
            return 45


def extra_perfoot_Raised(extra_Elevation):
    cost = 0

    if extra_Elevation > 2:
        if extra_Elevation > 8:
            cost += (1.05 * (extra_Elevation - 8))
            extra_Elevation = 8
        cost += (0.8 * (extra_Elevation - 2))

    return cost
