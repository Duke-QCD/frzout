# -*- coding: utf-8 -*-

from ..species import species_dict, _nth_digit, _normalize_species


def test_species():
    num = 9876543210

    assert [_nth_digit(num, n) for n in range(10)] == list(range(10)), \
        'Incorrect digits extracted from {}'.format(num)

    num = 7492

    assert [_nth_digit(num, n) for n in range(5, -1, -1)] == \
        [0, 0, 7, 4, 9, 2], \
        'Incorrect digits extracted from {}'.format(num)

    assert all(i > 100 for i in species_dict), \
        'Elementary particles in species data.'

    assert len(set(species_dict)) == len(species_dict), \
        'Duplicates in species data.'

    pion_data = species_dict[211]

    assert pion_data['name'] == 'pi', \
        'Incorrect pion name.'

    assert abs(pion_data['mass'] - .13957061) < 1e-15, \
        'Incorrect pion mass.'

    assert pion_data['has_anti'], \
        'The pi+ has an antiparticle.'

    assert pion_data['charge'] == 1, \
        'The pi+ has +1 charge.'

    assert pion_data['boson'], \
        'The pion is a boson.'

    assert not species_dict[111]['has_anti'], \
        'The pi0 does not have an antiparticle.'

    assert species_dict[311]['has_anti'], \
        'The K0 has an antiparticle.'

    assert not species_dict[2212]['boson'], \
        'The proton is a fermion.'

    assert 'mass_range' not in species_dict[211], \
        'The pion is stable.'

    assert 'mass_range' in species_dict[213], \
        'The rho is unstable.'

    assert abs(species_dict[213]['mass_range'][0] - .28) < 1e-12, \
        'The rho mass threshold is two pions.'

    assert all(
        i['mass_range'][0] >= .28
        for i in species_dict.values() if 'mass_range' in i
    ), 'The lightest decay product is two pions.'

    normalized = _normalize_species()

    assert len(normalized) == sum(
        2 if info['has_anti'] else 1 for info in species_dict.values()
    ), 'Incorrect number of normalized species.'

    assert normalized[0][0] == 111, \
        'First normalized species should be the pi0.'

    assert normalized[1][0] == -normalized[2][0] == 211, \
        'Second and third normalized species should be the pi+/-.'

    assert normalized[1][1] is normalized[2][1], \
        'pi+/- should share the same info dict.'

    normalized_id = _normalize_species('id')

    assert [i[0] for i in normalized_id] == [
        211, -211, 321, -321, 2212, -2212
    ], 'Incorrect normalized species IDs.'

    test_ID = 2112
    test_info = species_dict[test_ID]
    assert _normalize_species([test_ID]) == [
        (test_ID, test_info),
        (-test_ID, test_info)
    ], 'Incorrect custom normalized species.'
