"""Curated OpenDengue -> GADM 4.1 name maps for the dengue graph.

Geometry joins to series on NAMES, not codes, and OpenDengue geomatches against a different
shapefile lineage than GADM, so a residue of names never lines up on its own. `_canon` absorbs
accents, case and whitespace; everything here is genuine editorial difference, mapped by hand.
Every alias is an EXACT match against a real GADM key -- never fuzzy, which would bind
'antioquia|san andres' to 'andes', a different municipality. `validate_aliases()` asserts this.

There are TWO maps, and they run at different times:

  DENGUE_NAME_ALIASES  applied at LOAD, before the pivot. Repairs THE DATA.
  DENGUE_GADM_FIX      applied at the JOIN only. Absorbs GADM's own defects, and never
                       renames one of our nodes.

The load-time map must run before the pivot because it also REUNITES units whose series
OpenDengue split across two spellings in disjoint eras; those halves have to be summed before
the data are pivoted into a node-by-week matrix. See DENGUE_GADM_FIX for why the second map
cannot simply be folded into the first.

Shape: {country: {"parents": {data_adm1: gadm_adm1}, "units": {data_tail: gadm_tail}}}
  Admin1 country -> tail is 'adm1'          e.g. "nagasaki"
  Admin2 country -> tail is 'adm1|adm2'     e.g. "valle del cauca|cali"
`parents` is applied first, so `units` keys are written against the corrected parent name.

Curation loop: run the build, read the fail-loud diagnostic, add entries, repeat until zero
unmatched. A node is never dropped to make the join pass -- the only nodes that leave the graph
are the explicitly listed, reasoned entries in DENGUE_UNMAPPABLE.
"""

# Units created AFTER GADM 4.1's boundary vintage, so GADM has no polygon for the child. Its
# parent polygon still physically contains the child's territory (GADM predates the split), so
# folding the child's counts into the parent is geometrically exact and lossless. Listed here
# rather than buried in `units` so that each merge is individually auditable.
_PRESPLIT_MERGES = {
    "colombia": {
        "la guajira|albania": "la guajira|maicao",                 # Albania created 2000 ex-Maicao
        "bolivar|norosi": "bolivar|rio viejo",                     # Norosi ex-Rio Viejo
        "cauca|guachene": "cauca|caloto",                          # Guachene created 2006 ex-Caloto
        "cordoba|san jose de ure": "cordoba|montelibano",          # created 2007 ex-Montelibano
        "cordoba|tuchin": "cordoba|san andres de sotavento",       # created 2007 ex-S.A. Sotavento
    },
    "peru": {
        "loreto|datem del maranon": "loreto|alto amazonas",        # created 2005 ex-Alto Amazonas
        "loreto|putumayo": "loreto|maynas",                        # created 2015 ex-Maynas
    },
}

DENGUE_NAME_ALIASES: dict[str, dict[str, dict[str, str]]] = {

    # ---------------------------------------------------------------- brazil
    # 83 of 5,517 municipalities failed the join; all 27 states matched, so the residue is
    # purely leaf-level. 71 of them are OpenDengue stripping PUNCTUATION that GADM keeps --
    # apostrophes ("olho d agua" vs "olho d'agua") and hyphens ("xique xique" vs
    # "xique-xique"). Those 71 were resolved by an exact match on the punctuation-collapsed
    # name ([^a-z0-9] removed) *within the same state*, accepted only where the collapsed key
    # was UNIQUE in that state -- 0 were ambiguous. That is an exact match modulo punctuation,
    # not a fuzzy one, and the results are materialised here rather than left as a rule so the
    # committed map stays the single auditable artifact.
    "brazil": {"units": {
        # --- punctuation: apostrophes and hyphens GADM keeps and OpenDengue drops (71) ---
        'alagoas|olho d agua das flores': "alagoas|olho d'agua das flores",
        'alagoas|olho d agua do casado': "alagoas|olho d'agua do casado",
        'alagoas|olho d agua grande': "alagoas|olho d'agua grande",
        'alagoas|tanque d arca': "alagoas|tanque d'arca",
        'bahia|dias d avila': "bahia|dias d'avila",
        'bahia|xique xique': 'bahia|xique-xique',
        'goias|sao joao d alianca': "goias|sao joao d'alianca",
        'goias|sitio d abadia': "goias|sitio d'abadia",
        'maranhao|apicum acu': 'maranhao|apicum-acu',
        'maranhao|conceicao do lago acu': 'maranhao|conceicao do lago-acu',
        'maranhao|olho d agua das cunhas': "maranhao|olho d'agua das cunhas",
        'maranhao|pindare mirim': 'maranhao|pindare-mirim',
        'mato grosso|conquista d oeste': "mato grosso|conquista d'oeste",
        'mato grosso|figueiropolis d oeste': "mato grosso|figueiropolis d'oeste",
        'mato grosso|gloria d oeste': "mato grosso|gloria d'oeste",
        'mato grosso|lambari d oeste': "mato grosso|lambari d'oeste",
        'mato grosso|mirassol d oeste': "mato grosso|mirassol d'oeste",
        'minas gerais|guarda mor': 'minas gerais|guarda-mor',
        'minas gerais|olhos d agua': "minas gerais|olhos-d'agua",
        'minas gerais|pingo d agua': "minas gerais|pingo-d'agua",
        'minas gerais|sapucai mirim': 'minas gerais|sapucai-mirim',
        'minas gerais|sem peixe': 'minas gerais|sem-peixe',
        'paraiba|mae d agua': "paraiba|mae d'agua",
        'paraiba|olho d agua': "paraiba|olho d'agua",
        'parana|diamante d oeste': "parana|diamante d'oeste",
        'parana|itapejara d oeste': "parana|itapejara d'oeste",
        'parana|perola d oeste': "parana|perola d'oeste",
        'parana|rancho alegre d oeste': "parana|rancho alegre d'oeste",
        'parana|sao jorge d oeste': "parana|sao jorge d'oeste",
        'para|igarape acu': 'para|igarape-acu',
        'para|igarape miri': 'para|igarape-miri',
        'para|pau d arco': "para|pau d'arco",
        'para|peixe boi': 'para|peixe-boi',
        'para|tome acu': 'para|tome-acu',
        'piaui|barra d alcantara': "piaui|barra d'alcantara",
        'piaui|olho d agua do piaui': "piaui|olho d'agua do piaui",
        'piaui|pau d arco do piaui': "piaui|pau d'arco do piaui",
        'rio de janeiro|varre sai': 'rio de janeiro|varre-sai',
        'rio grande do norte|ceara mirim': 'rio grande do norte|ceara-mirim',
        'rio grande do norte|governador dix sept rosado':
            'rio grande do norte|governador dix-sept rosado',
        'rio grande do norte|lagoa d anta': "rio grande do norte|lagoa d'anta",
        'rio grande do norte|olho d agua do borges':
            "rio grande do norte|olho d'agua do borges",
        'rio grande do norte|venha ver': 'rio grande do norte|venha-ver',
        'rio grande do sul|entre ijuis': 'rio grande do sul|entre-ijuis',
        'rio grande do sul|nao me toque': 'rio grande do sul|nao-me-toque',
        'rio grande do sul|sant ana do livramento':
            "rio grande do sul|sant'ana do livramento",
        'rio grande do sul|xangri la': 'rio grande do sul|xangri-la',
        'rondonia|alta floresta d oeste': "rondonia|alta floresta d'oeste",
        'rondonia|alvorada d oeste': "rondonia|alvorada d'oeste",
        'rondonia|espigao d oeste': "rondonia|espigao d'oeste",
        'rondonia|guajara mirim': 'rondonia|guajara-mirim',
        'rondonia|ji parana': 'rondonia|ji-parana',
        'rondonia|machadinho d oeste': "rondonia|machadinho d'oeste",
        'rondonia|nova brasilandia d oeste': "rondonia|nova brasilandia d'oeste",
        'rondonia|santa luzia d oeste': "rondonia|santa luzia d'oeste",
        'rondonia|sao felipe d oeste': "rondonia|sao felipe d'oeste",
        'santa catarina|grao para': 'santa catarina|grao-para',
        'santa catarina|herval d oeste': "santa catarina|herval d'oeste",
        'sao paulo|aparecida d oeste': "sao paulo|aparecida d'oeste",
        'sao paulo|arco iris': 'sao paulo|arco-iris',
        'sao paulo|embu guacu': 'sao paulo|embu-guacu',
        'sao paulo|estrela d oeste': "sao paulo|estrela d'oeste",
        'sao paulo|guarani d oeste': "sao paulo|guarani d'oeste",
        'sao paulo|palmeira d oeste': "sao paulo|palmeira d'oeste",
        'sao paulo|pariquera acu': 'sao paulo|pariquera-acu',
        'sao paulo|santa barbara d oeste': "sao paulo|santa barbara d'oeste",
        'sao paulo|santa clara d oeste': "sao paulo|santa clara d'oeste",
        'sao paulo|santa rita d oeste': "sao paulo|santa rita d'oeste",
        'sao paulo|sao joao do pau d alho': "sao paulo|sao joao do pau d'alho",
        'sergipe|itaporanga d ajuda': "sergipe|itaporanga d'ajuda",
        'tocantins|pau d arco': "tocantins|pau d'arco",

        # --- genuine spelling variants: OpenDengue vs GADM/IBGE orthography (10) ---
        'bahia|muquem de sao francisco': 'bahia|muquem do sao francisco',
        'bahia|santa teresinha': 'bahia|santa terezinha',
        'ceara|itapage': 'ceara|itapaje',
        'mato grosso|poxoreo': 'mato grosso|poxoreu',
        'minas gerais|dona eusebia': 'minas gerais|dona euzebia',
        'minas gerais|sao thome das letras': 'minas gerais|sao tome das letras',
        'para|eldorado dos carajas': 'para|eldorado do carajas',
        'pernambuco|belem de sao francisco': 'pernambuco|belem do sao francisco',
        'sao paulo|florinia': 'sao paulo|florinea',
        'sao paulo|sao luis do paraitinga': 'sao paulo|sao luiz do paraitinga',

        # --- municipality RENAMES; the two sources sit on opposite sides of each (2) ---
        # Augusto Severo (RN) was renamed Campo Grande in 2013: GADM carries the NEW name,
        # OpenDengue the old. Tabocao (TO) was renamed Fortaleza do Tabocao: GADM carries the
        # OLD name, OpenDengue the new. Both are 1:1 renames, NOT merges -- verified that the
        # counterpart name is absent from the data in each case, so no node count changes.
        'rio grande do norte|augusto severo': 'rio grande do norte|campo grande',
        'tocantins|fortaleza do tabocao': 'tocantins|tabocao',
    }},

    # ---------------------------------------------------------------- argentina
    "argentina": {"units": {
        # Spanish numerals: OpenDengue writes digits, GADM spells them out.
        "chaco|12 de octubre": "chaco|doce de octubre",
        "chaco|25 de mayo": "chaco|veinticinco de mayo",
        "chaco|9 de julio": "chaco|nueve de julio",
        "misiones|25 de mayo": "misiones|veinticinco de mayo",
        "san juan|25 de mayo": "san juan|veinticinco de mayo",
        "santa fe|9 de julio": "santa fe|nueve de julio",
        # abbreviations
        "salta|grl. jose de san martin": "salta|general jose de san martin",
        "la rioja|general juan f. quiroga": "la rioja|general juan facundo quiroga",
        "santiago del estero|juan f. ibarra": "santiago del estero|juan felipe ibarra",
        # GADM splits the San Fernando partido into two polygons (mainland + Parana delta
        # islands); (1) is the mainland where essentially all reporting population lives.
        "buenos aires|san fernando": "buenos aires|san fernando (1)",
    }},

    # ---------------------------------------------------------------- colombia
    "colombia": {
        # Five department names orphaned 100 municipalities on their own.
        "parents": {
            "valle": "valle del cauca",
            "norte santander": "norte de santander",
            "guajira": "la guajira",
            "san andres": "san andres y providencia",
            "bogota": "bogota d.c.",
        },
        "units": {
            # '(cd)' = corregimiento departamental; GADM carries the bare name.
            "amazonas|el encanto (cd)": "amazonas|el encanto",
            "amazonas|la chorrera (cd)": "amazonas|la chorrera",
            "amazonas|la pedrera (cd)": "amazonas|la pedrera",
            "amazonas|la victoria (cd)": "amazonas|la victoria",
            "amazonas|miriti parana (cd)": "amazonas|miriti-parana",
            "amazonas|puerto alegria (cd)": "amazonas|puerto alegria",
            "amazonas|puerto arica (cd)": "amazonas|puerto arica",
            "amazonas|puerto santander (cd)": "amazonas|puerto santander",
            "amazonas|tarapaca (cd)": "amazonas|tarapaca",
            "guainia|barranco minas (cd)": "guainia|barranco minas",
            "guainia|cacahual (cd)": "guainia|cacahual",
            "guainia|la guadalupe (cd)": "guainia|la guadalupe",
            "guainia|mapiripana (cd)": "guainia|mapiripana",
            "guainia|pana pana (campo alegre) (cd)": "guainia|pana pana",
            "guainia|puerto colombia (cd)": "guainia|puerto colombia",
            "guainia|san felipe (cd)": "guainia|san felipe",
            "vaupes|pacoa (cd)": "vaupes|pacoa",
            "vaupes|yavarate (cd)": "vaupes|yavarate",
            # OpenDengue appends the municipal SEAT in parentheses; GADM does not. Note the
            # convention is not consistent -- sometimes the seat IS the GADM name (Robles ->
            # La Paz, Ospina Perez -> Venecia), so each was checked against GADM individually.
            "antioquia|puerto nare (la magdalena )": "antioquia|puerto nare",
            "antioquia|yondo (casabe)": "antioquia|yondo",
            "bolivar|tiquisio (puerto rico)": "bolivar|tiquisio",
            "cauca|patia (el bordo)": "cauca|patia",
            "cesar|robles (la paz)": "cesar|la paz",
            "choco|alto baudo (pie de pato)": "choco|alto baudo",
            "choco|bahia solano (mutis)": "choco|bahia solano",
            "choco|bajo baudo (pizarro)": "choco|bajo baudo",
            "choco|bojaya (bellavista)": "choco|bojaya",
            "choco|medio baudo (boca de pepe)": "choco|medio baudo",
            "cundinamarca|ospina perez (venecia)": "cundinamarca|venecia",
            "cundinamarca|rafael reyes (apulo)": "cundinamarca|rafael reyes",
            "huila|isnos (san jose de isnos)": "huila|isnos",
            "magdalena|ariguani (el dificil)": "magdalena|ariguani",
            "magdalena|pijino del carmen (pijino)": "magdalena|pijino del carmen",
            "narino|alban (san jose)": "narino|alban",
            "narino|arboleda (berruecos)": "narino|arboleda",
            "narino|colon (genova)": "narino|colon",
            "narino|francisco pizarro (salahonda)": "narino|francisco pizarro",
            "narino|los andes (sotomayor)": "narino|los andes",
            "narino|magui (payan)": "narino|magui",
            "narino|mallama (piedrancha)": "narino|mallama",
            "narino|olaya herrera(bocas de satinga": "narino|olaya herrera",
            "narino|roberto payan (san jose)": "narino|roberto payan",
            "narino|santa barbara (iscuande)": "narino|santa barbara",
            "narino|santa cruz (guachaves)": "narino|santa cruz",
            "putumayo|san miguel (la dorada)": "putumayo|san miguel",
            "sucre|coloso (ricaurte)": "sucre|coloso",
            "sucre|galeras (nueva granada)": "sucre|galeras",
            "tolima|armero (guayabal)": "tolima|armero",
            # spelling / spacing
            "atlantico|polo nuevo": "atlantico|polonuevo",
            "boyaca|cienega": "boyaca|cienaga",
            "cauca|villarica": "cauca|villa rica",
            "choco|itsmina": "choco|istmina",
            "choco|rioquito": "choco|rio quito",
            "magdalena|cerro san antonio": "magdalena|cerro de san antonio",
            "magdalena|puebloviejo": "magdalena|pueblo viejo",
            "magdalena|santa martha": "magdalena|santa marta",
            "magdalena|sitio nuevo": "magdalena|sitionuevo",
            "meta|vistahermosa": "meta|vista hermosa",
            "tolima|palocabildo": "tolima|palo cabildo",
            "tolima|villarica": "tolima|villarrica",
            "cordoba|san andres sotavento": "cordoba|san andres de sotavento",
            "cundinamarca|san antonio de tequendama": "cundinamarca|san antonio del tequendama",
            "santander|palmas socorro": "santander|palmas del socorro",
            # GADM uses the long OFFICIAL name where OpenDengue uses the common one.
            "antioquia|antioquia": "antioquia|santafe de antioquia",
            "antioquia|bolivar": "antioquia|ciudad bolivar",
            "antioquia|carmen de viboral": "antioquia|el carmen de viboral",
            "antioquia|carolina": "antioquia|carolina del principe",
            "antioquia|sopetran": "antioquia|el sopetran",
            "bolivar|cartagena": "bolivar|cartagena de indias",
            "bolivar|san estanislao": "bolivar|san estanislao de kostka",
            "cauca|lopez (micay)": "cauca|lopez de micay",
            "cesar|manaure balcon del cesar": "cesar|manaure",
            "choco|canton de san pablo (managru)": "choco|el canton del san pablo",
            "choco|carmen del darien": "choco|el carmen del darien",
            "choco|litoral del bajo san juan": "choco|el litoral del san juan",
            "cordoba|lorica": "cordoba|santa cruz de lorica",
            "cordoba|sahagun": "cordoba|san bernardino de sahagun",
            "cundinamarca|ubate": "cundinamarca|villa de san diego de ubate",
            "meta|cubarral": "meta|san luis de cubarral",
            "narino|el tablon": "narino|el tablon de gomez",
            "narino|pasto": "narino|san juan de pasto",
            "norte de santander|cucuta": "norte de santander|san jose de cucuta",
            "norte de santander|la playa": "norte de santander|la playa de belen",
            "norte de santander|salazar": "norte de santander|salazar de las palmas",
            "norte de santander|silos": "norte de santander|santo domingo de silos",
            "putumayo|mocoa": "putumayo|san miguel de mocoa",
            "sucre|la union": "sucre|la union de sucre",
            "tolima|mariquita": "tolima|san sebastian de mariquita",
            "valle del cauca|buga": "valle del cauca|guadalajara de buga",
            "valle del cauca|cali": "valle del cauca|santiago de cali",
            "bogota d.c.|bogota": "bogota d.c.|bogota d.c.",
            # Darien is the SEAT of the Calima municipality.
            "valle del cauca|darien": "valle del cauca|calima",
            # Sibling disambiguation, resolved from the data itself: OpenDengue carries
            # 'san pedro de uraba' and 'los palmitos' as SEPARATE nodes, so the bare names
            # must be the other member of each pair.
            "antioquia|san pedro": "antioquia|san pedro de los milagros",
            "sucre|palmito": "sucre|san antonio de palmito",
            "antioquia|san andres": "antioquia|san andres de cuerquia",
            # GADM mislabels Santander's 'San Andres' with Antioquia's suffix; the polygon
            # itself sits inside Santander, so the join is geometrically right.
            "santander|san andres": "santander|san andres de cuerquia",
        },
    },

    # ---------------------------------------------------------------- dominican republic
    "dominican republic": {"units": {
        "baoruco": "bahoruco",
        "el seibo": "el seybo",
        "elias pina": "la estrelleta",              # Elias Pina province = GADM's La Estrelleta
        # --- series-reuniting merges (disjoint eras; see module docstring) ---
        "hermanas mirabal": "salcedo",              # province renamed in 2007
        "maria trinidad sanches": "maria trinidad sanchez",
        "santiago. rodriguez": "santiago rodriguez",
    }},

    # ---------------------------------------------------------------- ecuador
    "ecuador": {"units": {
        "zamora chincipe": "zamora chinchipe",      # 2-week typo; reunites the series
    }},

    # ---------------------------------------------------------------- japan
    # (Japan needs no LOAD-time alias: its only mismatch was GADM's 'Naoasaki' typo, which is
    #  a defect in GADM, not in the data -- it now lives in DENGUE_GADM_FIX, applied at JOIN
    #  time, so our node id stays the correct `japan|nagasaki`.)

    # ---------------------------------------------------------------- nicaragua
    "nicaragua": {"units": {
        "bilwi": "atlantico norte",                 # Bilwi = capital of the RAAN
        "region autonoma del atlantico sur": "atlantico sur",
        # MINSA reports the Atlantico Sur department as TWO concurrent health districts
        # (SILAIS): 'RAAS' and 'Zelaya Central'. They partition the department, so summing
        # them reproduces the department total that GADM's single polygon represents.
        "zelaya central": "atlantico sur",
    }},

    # ---------------------------------------------------------------- peru
    # (Peru's 'Huenuco' GADM typo also moved to DENGUE_GADM_FIX -- see japan above.)
    "peru": {"units": {
        # OpenDengue encodes Maranon with a CYRILLIC SMALL LETTER IE (U+0435) where a Latin
        # 'e' belongs -- a homoglyph `_canon` cannot repair, so it is mapped explicitly.
        "huanuco|maraеon": "huanuco|maranon",
        # GADM splits Lima into the 'Lima' region and the 'Lima Province' metropolis.
        "lima|lima": "lima province|lima",
    }},

    # ---------------------------------------------------------------- taiwan
    # Taiwan's 287 township nodes are aggregated to their 22 parent counties (GADM 4.1 has
    # no Taiwanese township layer); these map the county names onto GADM's NAME_2.
    "taiwan": {"units": {
        "changhua county": "changhua", "hualien county": "hualien",
        "kaohsiung city": "kaohsiung", "keelung city": "keelung",
        "kinmen county": "kinmen", "miaoli county": "miaoli",
        "nantou county": "nantou", "new taipei city": "new taipei",
        "penghu county": "penghu", "pingtung county": "pingtung",
        "taichung city": "taichung", "tainan city": "tainan",
        "taipei city": "taipei", "taitung county": "taitung",
        "taoyuan city": "taoyuan", "yilan county": "yilan",
        "lienchiang county": "lienkiang",           # GADM romanises Lienchiang as 'Lienkiang'
        "yunlin county": "yulin",                   # GADM romanises Yunlin as 'Yulin'
        # chiayi/hsinchu keep their 'city'/'county' suffix in GADM and need no alias.
    }},
}

# Fold the pre-split merges into the unit maps.
for _c, _m in _PRESPLIT_MERGES.items():
    DENGUE_NAME_ALIASES.setdefault(_c, {}).setdefault("units", {}).update(_m)


# Nodes with NO GADM polygon and no defensible merge target. Listed explicitly, with a
# reason, so the drop is auditable -- the graph builder never drops a node on its own.
DENGUE_UNMAPPABLE: dict[str, dict[str, str]] = {
    "argentina": {
        # Formosa has exactly 9 departments and none is 'General Jose de San Martin' (that
        # department belongs to Chaco/Salta, both of which OpenDengue also reports). This is
        # a mislabelled source row; it cannot be reassigned without inventing data.
        "formosa|grl. jose de san martin":
            "no such department in Formosa (9 depts; belongs to Chaco/Salta) -- source error",
    },
}


# --------------------------------------------------------------------------- #
# GADM's own defects, absorbed at JOIN time so they never rename one of our nodes.
# --------------------------------------------------------------------------- #
# Keyed on our (correct) node-id tail -> the key GADM actually ships.
#
# This map is separate from DENGUE_NAME_ALIASES because a load-time alias rewrites the NODE ID.
# Folded into the load map, GADM's misspellings propagated into our own identifiers: the bundle
# shipped nodes called `japan|naoasaki` and `peru|huanuco|huenuco`, which would have been frozen
# into the released .npz and printed in the paper's node table.
#
# Flipping the direction of these entries in the load map does NOT work: that map runs before
# the pivot, so a reversed entry renames GADM's key rather than the data's and the join then
# misses the polygon. The separation of TIME is what fixes it, not the direction.
DENGUE_GADM_FIX: dict[str, dict[str, str]] = {
    "japan": {
        "nagasaki": "naoasaki",                 # GADM typo; the prefecture is Nagasaki
    },
    "peru": {
        # GADM typo in the Huanuco PROVINCE (the Admin1 region is spelt correctly, so only
        # the leaf differs).
        "huanuco|huanuco": "huanuco|huenuco",
    },
}


def validate_aliases(country_levels: dict, gadm_dir: str = "data/gadm") -> dict:
    """Check that every alias resolves to a real GADM key, guarding against typos in this file.
    Returns {country: [(src, bad_target), ...]}; an empty dict means the maps are sound.

    It is the EFFECTIVE join key that is checked -- DENGUE_GADM_FIX applied on top of the
    load-time target -- because a load-time target is a corrected NODE name, which need only be
    a GADM key after the join-time fix has run ('japan|nagasaki' is not in GADM; 'japan|naoasaki'
    is). Validating the load map against GADM directly would reject the two correct entries.
    """
    from to_schema import _gadm_gdf, _canon, GADM_ISO3
    bad: dict = {}
    for c in country_levels:
        keys = set(_gadm_gdf(gadm_dir, GADM_ISO3[c], country_levels[c], country=c)["_key"])
        gfix = {_canon(k): _canon(v) for k, v in (DENGUE_GADM_FIX.get(c) or {}).items()}

        # every load-time target, as the join will actually look it up
        for src, tgt in ((DENGUE_NAME_ALIASES.get(c) or {}).get("units") or {}).items():
            t = _canon(tgt)
            if gfix.get(t, t) not in keys:
                bad.setdefault(c, []).append((src, tgt))

        # and every join-time target must be a GADM key on its own
        for src, tgt in gfix.items():
            if tgt not in keys:
                bad.setdefault(c, []).append((src, tgt))
    return bad
