from datetime import timedelta

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Text,
    TextColumn,
)

from . import console


class SpeedColumn(ProgressColumn):
    def render(self, task) -> Text:
        speed = task.speed
        unit = "it/s"
        if speed is None:
            return Text(f"...{unit}", style="progress.data.speed")
        if speed < 1.0:
            speed, unit = 1 / speed, "s/it"
        return Text(f"{speed:.2f}{unit}", style="progress.data.speed")


class TimeColumn(ProgressColumn):
    def render(self, task) -> Text:
        # Calculate Elapsed
        elapsed = task.elapsed
        if elapsed is None:
            elapsed_string = "-:--:--"
        else:
            # Format using timedelta (removes microseconds)
            elapsed_string = str(timedelta(seconds=int(elapsed)))

        # Calculate Remaining
        remaining = task.time_remaining
        if remaining is None:
            remaining_string = "-:--:--"
        else:
            remaining_string = str(timedelta(seconds=int(remaining)))

        # Return the combined string
        return Text(
            f"{elapsed_string}<{remaining_string}",
            style="progress.remaining",
        )


def progress_bar(
    *,
    transient: bool = True,
    disable: bool = False,
    speed_estimate_period: float = 10800.0,
) -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        SpeedColumn(),
        TimeColumn(),
        console=console,
        speed_estimate_period=speed_estimate_period,
        transient=transient,
        disable=disable,
    )
