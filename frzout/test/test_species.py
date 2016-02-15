# -*- coding: utf-8 -*-

from ..species import species_dict, _nth_digit


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

    assert abs(pion_data['mass'] - .13957018) < 1e-15, \
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
