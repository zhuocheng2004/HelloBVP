import time


# Profiler
# used to record time usage of different part of the program

class Event:
    def __init__(self, name: str):
        self.name = name
        self.total_time = 0.0   # in seconds
        self.running = False
        self.last = 0
        self.count = 0

    def start(self):
        if self.running:
            raise Exception('already running:', self.name)

        self.running = True
        self.last = time.time()
        self.count += 1

    def end(self):
        if not self.running:
            raise Exception('not running:', self.name)

        self.running = False
        delta = time.time() - self.last
        self.total_time += delta

    def click(self):
        self.count += 1


events = {}


def get_event(name: str):
    if name in events:
        event = events[name]
    else:
        event = Event(name)
        events[name] = event

    return event


def reset():
    events.clear()


def push(name: str):
    event = get_event(name)
    event.start()


def pop(name: str):
    if name not in events:
        raise Exception('please push first:', name)

    event = events[name]
    event.end()


def click(name: str):
    event = get_event(name)
    event.click()


def get_time(name: str):
    if name not in events:
        raise Exception('no record:', name)

    return events[name].total_time


def get_count(name: str):
    if name not in events:
        raise Exception('no record:', name)

    return events[name].count


def report():
    print(' ======== Profiler Result (in seconds) ======== ')
    for event in events.values():
        print(' {:<24}  {:<8} {} '.format(event.name, f'[{event.count}]', event.total_time))
    print(' ======== END ======== ')
