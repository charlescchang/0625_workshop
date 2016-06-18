import sys
import copy
import numpy
import binascii
from datetime import datetime

def bytes_from_file(filename, chunksize=36800):
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                for b in chunk:
                    yield b
            else:
                print 'End'
                break

def load_dat(filename):

    begin_time = datetime.now()
    count = 0
    stat_count = 0
    total_stat_list = []
    total_next_list = []
    bstat_list = []
    wstat_list = []
    next_color = 0
    for b in bytes_from_file(filename):
        count +=1
        hex_str = binascii.b2a_hex(b)
        dec_int = int(hex_str,16)
        location = count % 368
        if location == 0:
            '''
            stat_count += 1
            if stat_count % 10000 == 0:
                end_time = datetime.now()
                elapse_time = end_time - begin_time
                print 'stat_count:' + str(stat_count)
                print 'elapse_time:' + str(elapse_time)
                begin_time = datetime.now()
            '''
            if next_move >= 0 and next_move < 361:
                if next_color == 1:
                  total_stat_list.append(copy.deepcopy(bstat_list)+copy.deepcopy(wstat_list))
                  total_next_list.append(next_move)
                elif next_color == 2:
                  total_stat_list.append(copy.deepcopy(wstat_list)+copy.deepcopy(bstat_list))
                  total_next_list.append(next_move)

            bstat_list = []
            wstat_list = []
            next_move = -1

        elif location > 0 and location <= 361:
            if dec_int == 1:
                bstat_list.append(1)
                wstat_list.append(0)
            elif dec_int == 2:
                bstat_list.append(0)
                wstat_list.append(1)
            else:
                bstat_list.append(0)
                wstat_list.append(0)

        elif location == 362:
            next_color = dec_int
        elif location == 365:
            next_move = dec_int
        elif location == 366:
            next_move = next_move + dec_int*256

    rval = (numpy.asarray(total_stat_list, dtype=float32), numpy.asarray(total_next_list, dtype=float32))
    return rval
