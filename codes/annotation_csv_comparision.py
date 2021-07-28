import os
import csv
import json
import sys


'SCRIPT TO COMPARE ANNOTATIONS OF DIFFERENT ANNOTATORS AND COMBINE THEM'


def csv_reader(csv_folder, csv_file):
    with open(os.path.join(csv_folder, csv_file), 'r') as c:
        reader = csv.reader(c, delimiter=',')

        for row in reader:
            yield row

def annotation_processing(annox_dict, interact_label_list, reason_tag_list, attributes_tag_list):
    # Interaction Label
    interact_label_list[int(annox_dict['Interaction Level'])] += 1

    # Reason Tags
    reason = list(annox_dict['Reason'].keys())
    reason = list(map(int, reason))  # gets the index of selected reasons

    # If participation attributes are present but not present in reason  [ACTIVATE THIS BASED ON HOW ANNO IS DONE]
    '''
    try:
        attributes = list(annox_dict['Attributes Present'].keys())
        attributes = list(map(int, attributes))  # gets the index of selected attributes

        # Check attributes row for active, passive and indifferent levels
        for x in attributes:
            if x == 2 or x == 3:
                reason.append(2)
            elif x == 7 or x == 8:
                reason.append(6)
            elif x == 13:
                reason.append(11)

    except KeyError:
        pass
    '''

    for r in list(set(reason)):
        reason_tag_list[r] += 1

    # reason_convert is a list of converted reason index to attribute index except for id 2 and 6
    reason_convert = [x for x in list(map(reason_mapping.get, reason)) if x is not None]

    # Attributes Tags
    try:
        attributes = list(annox_dict['Attributes Present'].keys())
        attributes = list(map(int, attributes))

    except KeyError:
        attributes = reason_convert

    final_attributes = list(set(reason_convert + attributes))  # Handles duplicates between reason and attributes
    final_attributes.sort()
    for a in final_attributes:
        attributes_tag_list[a] += 1


# Parameters
csv_folder = 'D:\Transfer_over\CSV_completed_annotation(egosocial)'   # change file here!
folder = 'test'
#csv_folder = 'D:\Interacction\Interaction_Dataset\Completed_Annotation'
#folder = 'zip29'
reader_list = []
num_of_labels = 3
mode = ['tie', 'proc']  # 'tie' to generate tie cases, 'proc' to process the annotations and save as 1 csv file
dataset = 'public'  # public or private

csv_file_list = [file for file in os.listdir(csv_folder) if file.__contains__(folder)]

for i in range(len(csv_file_list)):
    reader_list.append(csv_reader(csv_folder, csv_file_list[i]))

lists = [list(reader) for reader in reader_list]
print(csv_file_list)

tie_list = []

# mapping of reason -> attributes (reason_id -> attributes_id)
reason_mapping = dict([(0, 0), (1, 1), (3, 4), (4, 5), (5, 6), (7, 9), (8, 10), (9, 11), (10, 12), (11, 13)])

all_rows_reasons = []  # store all reason tags in list before dumping to csv
all_labels = []  # store all labels in list before dumping to csv
all_rows_attributes = []  # store all attribute tags in list before dumping to csv
episode = []  # store all episode ids in list before dumping to csv
person_id = []  # store all person ids in list before dumping to csv
sequence_id = []  # store all sequence ids in list before dumping to csv
face_crop_file = []  # store all face crop path in list before dumping to csv
all_rows_filename = []
all_rows_boundbox = []

for num, row in enumerate(zip(*lists)):  # For each big image

    if num > 0:
        interact_label_list = [0 for i in range(num_of_labels)]
        reason_tag_list = [0 for i in range(12)]
        attributes_tag_list = [0 for i in range(14)]

        # Checking Files for discrepancies
        # loop through different annotators, len(row) = number of annotators ie. row[0] is first annotator, row[1] is second annotator, ...
        for i in range(1, len(row)):
            try:
                assert row[0][0].split("\\")[-1] == row[i][0].split("\\")[-1]  # check filename

            except AssertionError:
                print("ERROR FILE for annotator:", i, row[i][0].split("\\")[-1])
                continue

            try:
                assert row[0][2] == row[i][2]  # check episode

            except AssertionError:
                print("ERROR EPISODE ID for annotator:", i, row[i][2])
                continue

            # face coordinates, person id, interaction level, reason, attributes present in annoX_dict
            anno1_dict = dict(json.loads(row[0][5]), **json.loads(row[0][6]))  # only row 5 and row 6 are important, first annotator
            anno2_dict = dict(json.loads(row[i][5]), **json.loads(row[i][6]))  # subsequent annotator

            if len(anno1_dict) != 0 and len(anno2_dict) != 0:
                try:
                    assert str(anno1_dict['person_id']) == str(anno2_dict['person_id'])  # check person_id

                except AssertionError:
                    print("ERROR PERSON ID for annotator:", i, "at line:", num+1, "expected:", anno1_dict['person_id'], "got:", anno2_dict['person_id'])
                    continue

                try:
                    if i == 1:  # For first annotator (executed only once)
                        annotation_processing(anno1_dict, interact_label_list, reason_tag_list, attributes_tag_list)
                        annotation_processing(anno2_dict, interact_label_list, reason_tag_list, attributes_tag_list)

                    else:  # For subsequent annotator
                        annotation_processing(anno2_dict, interact_label_list, reason_tag_list, attributes_tag_list)

                except KeyError:  # Annotator did not give Interaction Label or Reason or Both
                    try:
                        dummy = anno1_dict['Interaction Level']
                        print("ERROR Label for annotator", i, "at episode:", row[i][2], "for person:",
                              str(anno2_dict['person_id']), num + 1)
                        continue

                    except KeyError:
                        print("ERROR Label for annotator", 0, "at episode:", row[i][2], "for person:",
                              str(anno1_dict['person_id']), num + 1)
                        continue

        # label Processing
        if len(anno1_dict) != 0 and len(anno2_dict) != 0:  # file has faces to be annotated
            m = max(interact_label_list)
            max_index_list = [i for i, j in enumerate(interact_label_list) if j == m]

            if len(max_index_list) > 1:  # more than 1 max value for each label ie. A tie has occurred

                # [episode, person_id, H, L, N]
                tie_list.append([int(dict(json.loads(row[0][2]))["episode"]), str(anno1_dict['person_id']), *interact_label_list])
                continue

            else:
                final_label = interact_label_list.index(m)
                all_labels.append(final_label)
                #write to CSV

            avg_reason_tag = [x / len(row) for x in reason_tag_list]
            #print("Avg Reason Tag", avg_reason_tag, interact_label_list, final_label)
            all_rows_reasons.append(avg_reason_tag)

            avg_attributes_tag = [x / len(row) for x in attributes_tag_list]
            #print("Avg Attributes Tag", avg_attributes_tag, interact_label_list, final_label)
            all_rows_attributes.append(avg_attributes_tag)

            episode.append(row[0][2])
            person_id.append(anno1_dict['person_id'])
            sequence_id.append((str(str(dict(json.loads(row[0][2]))["episode"])) + "--" + str(anno1_dict['person_id'])))
            all_rows_filename.append(row[0][0][2:])
            all_rows_boundbox.append(row[0][5])

            if dataset == 'private':
                visited = False
                with open(os.path.join('D://Interacction/Interaction_Dataset', 'sequence1.csv'), 'r') as prev:
                    reader = csv.reader(prev, delimiter=',')
                    for line in reader:
                        # Check Date, Time, Person_ID, Episode_ID
                        if row[0][0][2:] == line[8]:
                            if anno1_dict['person_id'] == line[4].split('\\')[-1]:
                                #if str(dict(json.loads(row[0][2]))["episode"]) == str(line[6]):
                                face_crop_file.append(line[4] + '\\' + line[5])
                                visited = True
                                break
                        '''
                        if not len(all_rows_filename) == len(face_crop_file) + 1:
                            print(row[0][0][2:], anno1_dict['person_id'], str(dict(json.loads(row[0][2]))["episode"]))
                            break
                        '''
                    if visited is False:
                        print(row[0][0][2:])


for m in mode:
    if m == 'tie':
        # Find tie cases
        b_set = set(tuple(x) for x in tie_list)
        final_tie_list = [list(x) for x in b_set]
        final_tie_list.sort()
        print(final_tie_list)

        # Create "ties" folder
        if not os.path.exists(os.path.join(csv_folder, "tie_cases")):
            os.mkdir(os.path.join(csv_folder, "tie_cases"))

        # Save Tie cases to file
        with open(os.path.join(csv_folder, "tie_cases", f"{folder}_ties.txt"), 'w') as handle:
            for item in final_tie_list:
                ep = f'Episode_ID: {item[0]},'
                person = f'Person_ID: {item[1]},'
                hln = f'[H: {item[2]}, L: {item[3]}, N: {item[4]}]'
                handle.write('{} {} {}\n' .format(ep, person, hln))

    # TO GET INTO CSV
    if m == 'proc':
        if not os.path.exists(os.path.join(csv_folder, "processed_csv")):
            os.mkdir(os.path.join(csv_folder, "processed_csv"))

        if dataset == 'private':
            header = ['filename', 'bounding_box', 'face_crop', 'episode_id', 'person_id', 'sequence_id', 'final_label', 'reason_tag', 'attribute_tag']
        elif dataset == 'public':
            header = ['filename', 'bounding_box', 'episode_id', 'person_id', 'sequence_id', 'final_label',
                      'reason_tag', 'attribute_tag']
        with open(os.path.join(csv_folder, "processed_csv", f"{folder}_processed.csv"), 'w', newline='') as han:
            writer = csv.writer(han)
            writer.writerow(header)
            if dataset == 'private':
                for item in zip(all_rows_filename, all_rows_boundbox, face_crop_file, episode, person_id, sequence_id, all_labels, all_rows_reasons, all_rows_attributes):
                    writer.writerow(item)
            elif dataset == 'public':
                for item in zip(all_rows_filename, all_rows_boundbox, episode, person_id, sequence_id, all_labels, all_rows_reasons, all_rows_attributes):
                    writer.writerow(item)

