# import plotly.express as px
import subprocess


def plot_loss(l, skip=20):
    px.line({
        'iter': list(range(len(l)-skip)),
        'loss': l[skip:],
    }, x='iter', y='loss')


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()