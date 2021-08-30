import configuration


def all_vocabs_from_file(file):
    """
    Extract all features that may is contained into dataframe
    :param file: glossary text file of feature type
    :return: list of upper strings
    """
    return [line.split()[0].upper() for line in open(configuration.glossary_folder + file, 'r').readlines()]


_cases = all_vocabs_from_file('glossario_casos.txt')

_deaths = all_vocabs_from_file('glossario_mortes.txt')

features = {
    'cases': _cases,
    'deaths': _deaths
}
