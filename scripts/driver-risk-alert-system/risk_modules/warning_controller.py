from collections import defaultdict

last_warn_time = defaultdict(float)
yellow_warned = set()

def should_warn(track_id, now, level, score, stay_duration, config):
    score_thresholds = {
        'yellow': config['warning']['yellow_score_threshold'],
        'red': config['warning']['red_score_threshold']
    }

    if level == 'yellow':
        if score >= score_thresholds['yellow'] and track_id not in yellow_warned:
            yellow_warned.add(track_id)
            return True
        return False

    if level == 'red' and score >= score_thresholds['red']:
        interval = max(0.5, 2.0 - stay_duration * 0.1)
        if now - last_warn_time[track_id] > interval:
            last_warn_time[track_id] = now
            return True

    return False
