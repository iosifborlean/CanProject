import csv
import os
import re
import pandas as pd


def parse_logic_files(sample_rate, pre_padding_time, post_padding_time):
    SAMPLE_RATE = sample_rate * 1e6

    PRE_PADDING_TIME = pre_padding_time * 1e-6
    POST_PADDING_TIME = post_padding_time * 1e-6

    PRE_PADDING_SAMPLES = int(PRE_PADDING_TIME * SAMPLE_RATE)
    POST_PADDING_SAMPLES = int(POST_PADDING_TIME * SAMPLE_RATE)

    THRESHOLD = 2.2e-6
    MIN_DIFF = 1.6e-6

    OUTPUT_DIR = f"voltages/{sample_rate}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ANALOG_FILE = 'D:/analog.csv'
    DIGITAL_FILE = "D:/digital.csv"
    LOG = "D:/log.csv"

    # Carica digital.csv
    lines_digital = []
    with open(DIGITAL_FILE, newline='') as f1:
        reader = csv.reader(f1)
        next(reader)
        for row in reader:
            lines_digital.append(row)

    n_digital = len(lines_digital)

    # Carica le finestre CAN
    can_windows = []
    current_id = None
    current_id_start_time = None

    with open(LOG, newline='') as f2:
        reader = csv.reader(f2)
        next(reader)

        for row in reader:
            field_type = row[1].strip('"')
            start_time = float(row[2].strip('"'))

            if field_type == "identifier_field":
                identifier_full = row[4].strip('"')

                if identifier_full:
                    current_id = identifier_full[-3:]
                else:
                    current_id = None


            elif field_type == "control_field" or field_type == "data_field":

                if current_id is not None and current_id_start_time is None:
                    current_id_start_time = start_time


            elif field_type == "crc_field":
                if current_id is not None and current_id_start_time is not None:
                    duration = float(row[3].strip('"'))
                    window_end = start_time + duration
                    can_windows.append((current_id, current_id_start_time, window_end))
                    current_id = None
                    current_id_start_time = None

    n_windows = len(can_windows)

    id_file_counters = {}
    file_pattern = re.compile(r'^(.*?)_(\d+)\.csv$')
    for filename in os.listdir(OUTPUT_DIR):
        match = file_pattern.match(filename)
        if match:
            file_id = match.group(1)
            file_num = int(match.group(2))
            current_max = id_file_counters.get(file_id, 0)
            if file_num > current_max:
                id_file_counters[file_id] = file_num

    events_to_find = []
    window_idx = 0

    for i in range(n_digital - 1):
        current_line = lines_digital[i]
        can_l = current_line[1]

        if can_l != "0":
            continue

        next_line = lines_digital[i + 1]
        start = float(current_line[0])
        last_valid = float(next_line[0])
        diff = last_valid - start

        if diff > THRESHOLD or diff < MIN_DIFF:
            continue

        while window_idx < n_windows and can_windows[window_idx][2] < start:
            window_idx += 1

        matched_id = None
        if window_idx < n_windows:
            win_id, win_start, win_end = can_windows[window_idx]
            if win_start <= start <= win_end:
                matched_id = win_id

        if matched_id:
            events_to_find.append({
                'id': matched_id,
                'start': start,
                'end': last_valid,
                'found': False
            })

    print(f"Eventi da cercare: {len(events_to_find)}")
    print("Inizio elaborazione file analogico...")

    CHUNK_SIZE = 2000000
    MAX_SAMPLES_PER_ID = 1000
    analog_header = None
    skipped_voltage = 0
    skipped_boundary = 0
    id_sample_count = {}

    for chunk_num, chunk in enumerate(pd.read_csv(ANALOG_FILE, chunksize=CHUNK_SIZE)):

        if analog_header is None:
            analog_header = chunk.columns.tolist()

        time_col = chunk.columns[0]
        chunk_start_time = chunk[time_col].iloc[0]
        chunk_end_time = chunk[time_col].iloc[-1]

        print(f"Chunk {chunk_num}: {chunk_start_time:.6f} - {chunk_end_time:.6f} ({len(chunk)} righe)")

        for event in events_to_find:
            if event['found']:
                continue

            current_count = id_sample_count.get(event['id'], 0)
            if current_count >= MAX_SAMPLES_PER_ID:
                event['found'] = True
                continue

            event_window_start = event['start'] - PRE_PADDING_TIME
            event_window_end = event['end'] + POST_PADDING_TIME

            if event_window_end < chunk_start_time or event_window_start > chunk_end_time:
                continue

            if event_window_start < chunk_start_time or event_window_end > chunk_end_time:
                event['found'] = True
                skipped_boundary += 1
                continue

            mask = (chunk[time_col] >= event['start']) & (chunk[time_col] <= event['end'])
            core_samples = chunk[mask]

            if len(core_samples) == 0:
                continue

            first_idx = core_samples.index[0]
            last_idx = core_samples.index[-1]
            padded_start_idx = first_idx - PRE_PADDING_SAMPLES
            padded_end_idx = last_idx + POST_PADDING_SAMPLES

            segment = chunk.loc[padded_start_idx:padded_end_idx]
            segment_data = segment.values.tolist()

            core_samples_count = last_idx - first_idx + 1
            event_target_samples = core_samples_count + PRE_PADDING_SAMPLES + POST_PADDING_SAMPLES

            if len(segment_data) > event_target_samples:
                segment_data = segment_data[:event_target_samples]
            elif len(segment_data) < event_target_samples:
                num_to_pad = event_target_samples - len(segment_data)
                segment_data.extend([segment_data[-1]] * num_to_pad)

            if len(segment_data) != event_target_samples:
                continue

            # Index 1 is Can H, Index 2 is Can L
            first_v_diff = abs(segment_data[0][1] - segment_data[0][2])
            last_v_diff = abs(segment_data[-1][1] - segment_data[-1][2])

            MAX_RECESSIVE_DIFF = 0.5

            if first_v_diff > MAX_RECESSIVE_DIFF or last_v_diff > MAX_RECESSIVE_DIFF:
                print(
                    f"ID {event['id']} skipped: Boundary not recessive. Vdiff_start={first_v_diff:.2f}V, Vdiff_end={last_v_diff:.2f}V")
                # DO NOT set event['found'] = True here if you want to retry it,
                # but if you want to drop it, just look at the current chunk setup:
                event['found'] = True
                skipped_voltage += 1
                continue

            current_count = id_file_counters.get(event['id'], 0) + 1
            id_file_counters[event['id']] = current_count
            id_sample_count[event['id']] = id_sample_count.get(event['id'], 0) + 1

            filename = os.path.join(OUTPUT_DIR, f"{event['id']}_{current_count}.csv")

            with open(filename, 'w', newline='') as f_out:
                writer = csv.writer(f_out)
                writer.writerow(analog_header)
                writer.writerows(segment_data)

            print(
                f"ID {event['id']}, salvato {filename} (V_start: {first_v_diff:.2f}V, V_end: {last_v_diff:.2f}V) [{id_sample_count[event['id']]}/{MAX_SAMPLES_PER_ID}]")
            event['found'] = True

        if all(e['found'] for e in events_to_find):
            print("Tutti gli eventi trovati!")
            break

    print(f"\nFrequenza ID: {id_file_counters}")
    print(f"Eventi trovati: {sum(1 for e in events_to_find if e['found'])}/{len(events_to_find)}")
    print(f"File skippati per voltaggio: {skipped_voltage}")
    print(f"Eventi scartati (a cavallo di due chunk): {skipped_boundary}")


parse_logic_files(50, 0.7, 0.6)