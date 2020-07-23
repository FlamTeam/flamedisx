#########################################################################
#
# Script name: test_write.py
# Purpose: Creating dummy maps of ones and zeros on the typical domains of S1
# and S2 signals.
#
# Pueh Leng Tan, 23 July 2020

import json

sig_type = ['s1', 's2']
val_type = ['zeros', 'ones']

for loc_sig in sig_type:
    if loc_sig=='s1':
        fread = 'ReconstructionS1BiasMeanLowers_SR1_v2.json'
    elif loc_sig=='s2':
        fread = 'ReconstructionS2BiasMeanLowers_SR1_v2.json'
    else:
        print('FATAL: Oi. Only s1 or s2 la pls, cmmon.')
        raise

    for loc_val in val_type:
        fwrite = 'dummy_%s_%s.json' % (loc_val, loc_sig)
        description = 'Map of just %s on typical %s domain' % (loc_val, loc_sig)

        if loc_val=='ones':
            val = [1.]
        elif loc_val=='zeros':
            val = [0.]
        else:
            print('FATAL: Oi. Only ones or zeros la pls, cmmon.')
            raise

        with open(fread) as in_fid:
            tmp = json.load(in_fid)
            tmp['time'] = 1595512560.4242424
            tmp['map'] = val*len(tmp['map'])
            tmp['name'] = fwrite[:-5]
            tmp['description'] = description
            tmp['coordinate_system'][0][0] = loc_sig

# only write if file doesn't already exists
        with open(fwrite, 'x') as out_fid:
            json.dump(tmp, out_fid)
