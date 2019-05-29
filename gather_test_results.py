import os
import subprocess
from collections import defaultdict
import statistics as st


def get_statistics(times_ns):
    diffs = [(item[1] - item[0]) / 1000 for item in list(zip(times_ns, times_ns[1:]))]
    return get_statistics_from_diffs(diffs)


def get_statistics_from_diffs(diffs):
    the_mean = st.mean(diffs)
    return diffs, {'min': min(diffs), 'max': max(diffs),
                   'mean': the_mean, 'median': st.median(diffs), 'stdev': st.stdev(diffs, the_mean)}


# stops at three newlines
def get_multi_line_input(first_string):
    input_list = [input(first_string)]
    while True:
        line = input().strip()
        if line == "" and input_list[-1] == "":
            break
        input_list.append(line)

    return "\n".join(input_list[:-1])


def read_until_empty(file, a_type):
    items = []
    for a_line in file:
        a_line = a_line.strip()
        if a_line != "":
            items.append(a_type(a_line))
        else:
            break
    return items


def ssh_and_execute_commands(use_vagrant, path, commands, get_output=True):
    if use_vagrant:
        command = f"cd {path}; "\
                  f"vagrant ssh -c \"{commands}\""
    else:
        command = f"ssh -tt {path} \"{commands}\""
    if get_output:
        return subprocess.check_output(command, encoding='UTF-8', shell=True)
    else:
        subprocess.run(command, shell=True)


def read_app_server_logs(use_vagrant, path):
    print("> Reading app logs.. ", end="")
    sub_command = "cat /var/log/uwsgi/reqlog;" \
                  "echo -e '\n\n';" \
                  "cat /var/log/uwsgi/errlog;"
    result = ssh_and_execute_commands(use_vagrant, path, sub_command)
    print("Done.")
    return result


def restart_app_and_clear_logs(use_vagrant, path):
    print("> Restarting app and clearing logs.. ", end="")
    sub_command = "sudo systemctl restart app; " \
                  "echo '' > /var/log/uwsgi/reqlog; " \
                  "echo '' > /var/log/uwsgi/errlog; " \
                  "exit;"
    ssh_and_execute_commands(use_vagrant, path, sub_command, False)
    print("Done.")


def start_recording_traffic(device, path, duration):
    print(f"> Recording traffic for {duration} seconds..")
    command = f"tshark -i {device} -a duration:{duration} -w {path}"
    subprocess.run(command, shell=True)
    print("Done.")


def filter_and_parse_pcap(path, tcp_port):
    print(f"> Parsing and filtering pcap..")
    command = f"tcpdump -r {path} -w - 'tcp dst port {tcp_port} and tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x504f5354' | " \
              "tshark -r - -i 2 -T fields -e frame.time_epoch"
    result = subprocess.check_output(command, encoding='UTF-8', shell=True)
    print("Done.")
    return result


def get_results(use_vagrant, path, num_duplicate_tests=15, recording_duration=8):
    only_statistics = False

    test_results_base = "results/"
    test_name = input("Provide input <name> <type> (type = f/n/s) ")
    filename = test_name.replace(" ", "_")
    test_results_base += test_name.split(" ")[0] + "/"
    if not os.path.exists(test_results_base):
        os.mkdir(test_results_base)
    test_results_base += test_name.split(" ")[1] + "/"
    if not os.path.exists(test_results_base):
        os.mkdir(test_results_base)
    test_results_location = test_results_base + filename

    # read test type
    device = "lo0"
    if use_vagrant:
        test_type = test_name.split(" ")[1][0]
        if test_type == "f":
            port = 5005
        elif test_type == "n":
            port = 5006
        else:  # test_type == "s":
            port = 5007
    else:
        port = 80
        device = "en0"

    all_statistics = defaultdict(list)
    app_is_restarted = False
    for i in range(1, num_duplicate_tests + 1):
        test_results_location_i = f"{test_results_location}_{i}"

        local_times = []
        web_server_times = []
        app_server_times = []
        # check if duplicate result already exists
        if os.path.exists(f"{test_results_location_i}_in.txt"):
            # read input file
            print(f"\n> Found result input {i} -> skip creation")
            with open(test_results_location_i + "_in.txt", 'r') as file:
                local_times = read_until_empty(file, float)
                web_server_times = read_until_empty(file, float)
                app_server_times = read_until_empty(file, float)
                total_num = int(next(file, 0))
                success_num = int(next(file, 0))
                voucher_used_num = int(next(file, 0))
        elif not only_statistics:
            # create input file
            print(f"\n> Did not find result input {i} -> perform new test")
            if not app_is_restarted:
                restart_app_and_clear_logs(use_vagrant, path)

            input(f"> Press ENTER, wait for text: \"Capturing on '(..)'\" and then send all parallel requests.\n")
            start_recording_traffic(device, test_results_location_i + "_rec.pcapng", recording_duration)
            app_is_restarted = False

            log_data = read_app_server_logs(use_vagrant, path)
            pcap_data = filter_and_parse_pcap(test_results_location_i + "_rec.pcapng", port)

            # parse pcap data
            for item in pcap_data.split("\n"):
                if item:
                    time_ns = float(item.strip()) * 1e6
                    local_times.append(time_ns)
            local_times = sorted(local_times)

            # parse log data (web server)
            for line in log_data.split("\n\n")[0].split("\n"):
                line = line.strip()
                if line:
                    time_ns = float(line)
                    web_server_times.append(time_ns)
            web_server_times = sorted(web_server_times)

            # parse log data (app server)
            for line in log_data.split("\n\n")[-1].split("\n"):
                if 'app - DEBUG' in line:
                    time_ns = float(line.split(" ")[-1].strip())
                    app_server_times.append(time_ns)
            app_server_times = sorted(app_server_times)

            if len(local_times) == 0:
                print("ERR: No client results! Skipping (retry it in next run)")
                input("> Press any key to continue..")
                continue
            elif len(web_server_times) == 0 or len(app_server_times) == 0:
                print("ERR: No server results! Skipping (retry it in next run)")
                input("> Press any key to continue..")
                continue

            # restart app if the tool waits on gateway timeouts
            while True:
                answer = str(input("> Tool done? (ENTER = Yes, 'n' = No)"))
                if answer.strip() == "n":
                    restart_app_and_clear_logs(use_vagrant, path)
                    app_is_restarted = True
                    break
                elif answer.strip() == "":
                    break
                print("Failed to parse input!")

            # input test results
            total_num = 25
            success_num = -1
            while success_num < 0:
                try:
                    success_num = int(input("> Results: total 20X return codes? "))
                except Exception:
                    print("Failed to parse input!")
            voucher_used_num = -1
            while voucher_used_num < 0:
                try:
                    voucher_used_num = int(input("> Results: total vouchers used? "))
                except Exception:
                    print("Failed to parse input!")

            print(f"> Writing results input file.. ", end="")
            with open(f"{test_results_location}_{i}_in.txt", "w") as file:
                file.write("\n".join([str(item) for item in local_times]) + "\n\n")
                file.write("\n".join([str(item) for item in web_server_times]) + "\n\n")
                file.write("\n".join([str(item) for item in app_server_times]) + "\n\n")
                file.write(f"{total_num}\n")
                file.write(f"{success_num}\n")
                file.write(f"{voucher_used_num}\n")
            print("Done.")

        # calculate statistics
        local_diff, local = get_statistics(local_times)
        web_diff, web = get_statistics(web_server_times)
        app_diff, app = get_statistics(app_server_times)

        all_statistics['local'].extend(local_diff)
        all_statistics['web'].extend(web_diff)
        all_statistics['app'].extend(app_diff)
        all_statistics['success'].append(success_num)
        all_statistics['vouchers_used'].append(voucher_used_num)
        diffs = (st.mean(app_server_times) - st.mean(web_server_times)) / 1000
        all_statistics['diffs'].append(diffs)

        if local_times and not os.path.exists(f"{test_results_location_i}_out.txt"):
            print(f"> Writing results statistics file.. ", end="")

            out = ""
            out += "Time-differences (millis):"
            out += "\n\n" + "\n".join([str(item) for item in local_diff])
            out += "\n\n" + "\n".join([str(item) for item in web_diff])
            out += "\n\n" + "\n".join([str(item) for item in app_diff])

            out += f"\n\nTest:\t{filename}_{i}"
            out += f"\nRequests:\t{total_num}"
            out += f"\nSuccess:\t{success_num}"
            out += f"\nUsed:   \t{voucher_used_num}"

            out += "\n\nLocal times (milisec):\n"
            out += "\n".join([f"{item[0]}:\t{item[1]}" for item in local.items()])

            out += "\n\nWeb server times (milisec):\n"
            out += "\n".join([f"{item[0]}:\t{item[1]}" for item in web.items()])

            out += "\n\nApp server times (milisec):\n"
            out += "\n".join([f"{item[0]}:\t{item[1]}" for item in app.items()])
            out += "\ndiff:\t" + str(diffs)

            with open(f"{test_results_location_i}_out.txt", "w") as file:
                file.write(out)
            print("Done.")
        else:
            print(f"> Found result output {i} -> skip calculation")

    if all_statistics['local'] and (not os.path.exists(f"{test_results_location}_totals.txt")
                                    or not os.path.exists(f"{test_results_location}_diffs.txt")):

        local_diff, local = get_statistics_from_diffs(all_statistics['local'])
        web_diff, web = get_statistics_from_diffs(all_statistics['web'])
        app_diff, app = get_statistics_from_diffs(all_statistics['app'])
        diffs_diff, diffs = get_statistics_from_diffs(all_statistics['diffs'])
        success_diff, suc = get_statistics_from_diffs(all_statistics['success'])
        vouchers_diff, vouchers = get_statistics_from_diffs(all_statistics['vouchers_used'])
        ratio_diff, ratio = get_statistics_from_diffs(
            [item[0] / item[1] for item in zip(all_statistics['success'], all_statistics['vouchers_used'])])

        print(f"\n> Writing total results statistics file.. ", end="")

        out = ""
        out += f"Test:\t{filename}"
        out += f"\n\nResults:\t{len(all_statistics['success'])}"

        out += "\n\nSuccess codes:\n"
        out += "\n".join([f"{item[0]}:\t{item[1]}" for item in suc.items()])

        out += "\n\nVouchers used:\n"
        out += "\n".join([f"{item[0]}:\t{item[1]}" for item in vouchers.items()])

        out += "\n\nRatio (successes / used vouchers):\n"
        out += "\n".join([f"{item[0]}:\t{item[1]}" for item in ratio.items()])

        out += "\n\nLocal times (milisec):\n"
        out += "\n".join([f"{item[0]}:\t{item[1]}" for item in local.items()])

        out += "\n\nWeb server times (milisec):\n"
        out += "\n".join([f"{item[0]}:\t{item[1]}" for item in web.items()])

        out += "\n\nApp server times (milisec):\n"
        out += "\n".join([f"{item[0]}:\t{item[1]}" for item in app.items()])

        out += "\n\nDiff between web and app server (milisec):\n"
        out += "\n".join([f"{item[0]}:\t{item[1]}" for item in diffs.items()])

        with open(f"{test_results_location}_totals.txt", "w") as file:
            file.write(out)
        with open(f"{test_results_location}_diffs.txt", "w") as file:
            file.write("Success codes:\n" + "\n".join([str(item) for item in success_diff]))
            file.write("\n\nVouchers used:\n" + "\n".join([str(item) for item in vouchers_diff]))
            file.write("\n\nRatio:\n" + "\n".join([str(item) for item in ratio_diff]))
            file.write("\n\nLocal:\n" + "\n".join([str(item) for item in local_diff]))
            file.write("\n\nWeb:\n" + "\n".join([str(item) for item in web_diff]))
            file.write("\n\nApp:\n" + "\n".join([str(item) for item in app_diff]))
            file.write("\n\nDiff:\n" + "\n".join([str(item) for item in diffs_diff]))
        print("Done.")
    else:
        print(f"\n> Found total result output -> skip calculation")


def fix_names():
    command = "find . -name \"*.txt\" -o -name \"*.pcapng\" "
    result_file_list = subprocess.check_output(command, encoding='UTF-8', shell=True)
    for result_file in result_file_list.split("\n"):
        if not result_file:
            continue
        parts = result_file.split("/")
        if parts[2] != parts[4].split("_")[0]:
            os.rename(result_file, result_file.replace(parts[4].split("_")[0] + "_", parts[2] + "_"))


test_app_location = "/Users/rvanemous/Documents/Afstuderen/Tools/CompuRacer/TestWebAppVouchers/app/"

get_results(False, "ubuntu@52.47.174.65", 25, 30)
# fix_names()
