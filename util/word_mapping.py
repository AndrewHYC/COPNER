from collections import OrderedDict

# # Few-NERD
FEWNERD_WORD_MAP = OrderedDict()

FEWNERD_WORD_MAP['O'] = 'none'

FEWNERD_WORD_MAP['location-GPE'] = 'nation'
FEWNERD_WORD_MAP['location-bodiesofwater'] = 'water'
FEWNERD_WORD_MAP['location-island'] = 'island'
FEWNERD_WORD_MAP['location-mountain'] = 'mountain'
FEWNERD_WORD_MAP['location-park'] = 'parks'
FEWNERD_WORD_MAP['location-road/railway/highway/transit'] = 'road'
FEWNERD_WORD_MAP['location-other'] = 'location'

FEWNERD_WORD_MAP['person-actor'] = 'actor'
FEWNERD_WORD_MAP['person-artist/author'] = 'artist'
FEWNERD_WORD_MAP['person-athlete'] = 'athlete'
FEWNERD_WORD_MAP['person-director'] = 'director'
FEWNERD_WORD_MAP['person-politician'] = 'politician'
FEWNERD_WORD_MAP['person-scholar'] = 'scholar'
FEWNERD_WORD_MAP['person-soldier'] = 'soldier'
FEWNERD_WORD_MAP['person-other'] = 'person'

FEWNERD_WORD_MAP['organization-company'] = 'company'
FEWNERD_WORD_MAP['organization-education'] = 'education'
FEWNERD_WORD_MAP['organization-government/governmentagency'] = 'government'
FEWNERD_WORD_MAP['organization-media/newspaper'] = 'media'
FEWNERD_WORD_MAP['organization-politicalparty'] = 'parties'
FEWNERD_WORD_MAP['organization-religion'] = 'religion'
FEWNERD_WORD_MAP['organization-showorganization'] = 'show'
FEWNERD_WORD_MAP['organization-sportsleague'] = 'league'
FEWNERD_WORD_MAP['organization-sportsteam'] = 'team'
FEWNERD_WORD_MAP['organization-other'] = 'organization'

FEWNERD_WORD_MAP['building-airport'] = 'airport'
FEWNERD_WORD_MAP['building-hospital'] = 'hospital'
FEWNERD_WORD_MAP['building-hotel'] = 'hotel'
FEWNERD_WORD_MAP['building-library'] = 'library'
FEWNERD_WORD_MAP['building-restaurant'] = 'restaurant'
FEWNERD_WORD_MAP['building-sportsfacility'] = 'facility'
FEWNERD_WORD_MAP['building-theater'] = 'theater'
FEWNERD_WORD_MAP['building-other'] = 'building'

FEWNERD_WORD_MAP['art-broadcastprogram'] = 'broadcast'
FEWNERD_WORD_MAP['art-film'] = 'film'
FEWNERD_WORD_MAP['art-music'] = 'music'
FEWNERD_WORD_MAP['art-painting'] = 'painting'
FEWNERD_WORD_MAP['art-writtenart'] = 'writing'
FEWNERD_WORD_MAP['art-other'] = 'art'

FEWNERD_WORD_MAP['product-airplane'] = 'airplane'
FEWNERD_WORD_MAP['product-car'] = 'car'
FEWNERD_WORD_MAP['product-food'] = 'food'
FEWNERD_WORD_MAP['product-game'] = 'game'
FEWNERD_WORD_MAP['product-ship'] = 'ship'
FEWNERD_WORD_MAP['product-software'] = 'software'
FEWNERD_WORD_MAP['product-train'] = 'train'
FEWNERD_WORD_MAP['product-weapon'] = 'weapon'
FEWNERD_WORD_MAP['product-other'] = 'product'

FEWNERD_WORD_MAP['event-attack/battle/war/militaryconflict'] = 'war'
FEWNERD_WORD_MAP['event-disaster'] = 'disaster'
FEWNERD_WORD_MAP['event-election'] = 'election'
FEWNERD_WORD_MAP['event-protest'] = 'protest'
FEWNERD_WORD_MAP['event-sportsevent'] = 'sport'
FEWNERD_WORD_MAP['event-other'] = 'event'

FEWNERD_WORD_MAP['other-astronomything'] = 'astronomy'
FEWNERD_WORD_MAP['other-award'] = 'award'
FEWNERD_WORD_MAP['other-biologything'] = 'biology'
FEWNERD_WORD_MAP['other-chemicalthing'] = 'chemistry'
FEWNERD_WORD_MAP['other-currency'] = 'currency'
FEWNERD_WORD_MAP['other-disease'] = 'disease'
FEWNERD_WORD_MAP['other-educationaldegree'] = 'degree'
FEWNERD_WORD_MAP['other-god'] = 'god'
FEWNERD_WORD_MAP['other-language'] = 'language'
FEWNERD_WORD_MAP['other-law'] = 'law'
FEWNERD_WORD_MAP['other-livingthing'] = 'organism'
FEWNERD_WORD_MAP['other-medical'] = 'medical'


# # OntoNotes
ONTONOTES_WORD_MAP = OrderedDict()

ONTONOTES_WORD_MAP['O'] = 'none'

ONTONOTES_WORD_MAP['ORG'] = 'organization'
ONTONOTES_WORD_MAP['NORP'] = 'country'
ONTONOTES_WORD_MAP['ORDINAL'] = 'number'
ONTONOTES_WORD_MAP['WORK_OF_ART'] = 'art'
ONTONOTES_WORD_MAP['QUANTITY'] = 'quantity'
ONTONOTES_WORD_MAP['LAW'] = 'law'

ONTONOTES_WORD_MAP['GPE'] = 'nation'
ONTONOTES_WORD_MAP['CARDINAL'] = 'cardinal'
ONTONOTES_WORD_MAP['PERCENT'] = 'percent'
ONTONOTES_WORD_MAP['TIME'] = 'time'
ONTONOTES_WORD_MAP['EVENT'] = 'event'
ONTONOTES_WORD_MAP['LANGUAGE'] = 'language'

ONTONOTES_WORD_MAP['PERSON'] = 'person'
ONTONOTES_WORD_MAP['DATE'] = 'date'
ONTONOTES_WORD_MAP['MONEY'] = 'money'
ONTONOTES_WORD_MAP['LOC'] = 'location'
ONTONOTES_WORD_MAP['FAC'] = 'facility'
ONTONOTES_WORD_MAP['PRODUCT'] = 'product'


# # CoNLL 03
CONLL_WORD_MAP = OrderedDict()

CONLL_WORD_MAP['O'] = 'none'

CONLL_WORD_MAP['ORG'] = 'organization'
CONLL_WORD_MAP['PER'] = 'person'
CONLL_WORD_MAP['LOC'] = 'location'
CONLL_WORD_MAP['MISC'] = 'miscellaneous'


# # WNUT
WNUT_WORD_MAP = OrderedDict()

WNUT_WORD_MAP['O'] = 'none'

WNUT_WORD_MAP['location'] = 'location'
WNUT_WORD_MAP['group'] = 'group'
WNUT_WORD_MAP['corporation'] = 'company'
WNUT_WORD_MAP['person'] = 'person'
WNUT_WORD_MAP['creative-work'] = 'creativity'
WNUT_WORD_MAP['product'] = 'product'
                            

# # I2B2
I2B2_WORD_MAP = OrderedDict()

I2B2_WORD_MAP['O'] = 'none'

I2B2_WORD_MAP['DATE'] = 'date' #
I2B2_WORD_MAP['AGE'] = 'age' #
I2B2_WORD_MAP['STATE'] = 'state' #
I2B2_WORD_MAP['PATIENT'] = 'people'
I2B2_WORD_MAP['DOCTOR'] = 'doctor'
I2B2_WORD_MAP['MEDICALRECORD'] = 'number'
I2B2_WORD_MAP['HOSPITAL'] = 'hospital'
I2B2_WORD_MAP['PHONE'] = 'phone'
I2B2_WORD_MAP['IDNUM'] = 'id'
I2B2_WORD_MAP['USERNAME'] = 'name'

I2B2_WORD_MAP['STREET'] = 'street'
I2B2_WORD_MAP['CITY'] = 'city'
I2B2_WORD_MAP['ZIP'] = 'zip'
I2B2_WORD_MAP['EMAIL'] = 'email'
I2B2_WORD_MAP['PROFESSION'] = 'profession'
I2B2_WORD_MAP['COUNTRY'] = 'country'
I2B2_WORD_MAP['ORGANIZATION'] = 'organization'
I2B2_WORD_MAP['FAX'] = ['ip']
I2B2_WORD_MAP['LOCATION-OTHER'] = 'location' #=
I2B2_WORD_MAP['DEVICE'] = 'device' #=

I2B2_WORD_MAP['BIOID'] = 'bio'
I2B2_WORD_MAP['HEALTHPLAN'] = 'plan'
I2B2_WORD_MAP['URL'] = 'link'

# # MIT-Movies
MOVIES_WORD_MAP = OrderedDict()

MOVIES_WORD_MAP['O'] = 'none'

MOVIES_WORD_MAP['CHARACTER'] = 'character'
MOVIES_WORD_MAP['GENRE'] = 'genre'
MOVIES_WORD_MAP['TITLE'] = 'title'
MOVIES_WORD_MAP['PLOT'] = 'plot'
MOVIES_WORD_MAP['RATING'] = 'rating'
MOVIES_WORD_MAP['YEAR'] = 'year'

MOVIES_WORD_MAP['REVIEW'] = 'review'
MOVIES_WORD_MAP['ACTOR'] = 'actor'
MOVIES_WORD_MAP['DIRECTOR'] = 'director'
MOVIES_WORD_MAP['SONG'] = 'song'
MOVIES_WORD_MAP['RATINGS_AVERAGE'] = 'average'
MOVIES_WORD_MAP['TRAILER'] = 'trailer'

