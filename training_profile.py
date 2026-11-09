"""Write standard pstats files and a human-readable function ranking."""
import pstats


def save_profile(profiler, folder, name):
    profiler.disable()
    profiler.dump_stats(str(folder / f'{name}.prof'))
    with (folder / f'{name}.txt').open('w', encoding='utf-8') as stream:
        stats = pstats.Stats(profiler, stream=stream).strip_dirs()
        for order in ('tottime', 'cumtime'):
            stats.sort_stats(order).print_stats(35)
