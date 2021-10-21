require(["nbextensions/snippets_menu/main"], function (snippets_menu) {
    console.log('Loading `snippets_menu` customizations from `custom.js`');

    var magics = {//{{{
        'name': 'Magics',
        'sub-menu-direction': 'left',
        'sub-menu' : [
            {
                'name': '_IMPORT',//{{{
                'snippet' : [
                    "%reload_ext watermark",
                    "%reload_ext autoreload",
                    "%autoreload 2",
                    "%watermark -p numpy,sklearn,pandas",
                    "%watermark -p ipywidgets,cv2,PIL,matplotlib,plotly,netron",
                    "%watermark -p torch,torchvision,torchaudio",
                    "%watermark -p tensorflow,tensorboard,tflite",
                    "%watermark -p onnx,tf2onnx,onnxruntime,tensorrt,tvm",
                    "%matplotlib inline",
                    "%config InlineBackend.figure_format='retina'",
                    "%config IPCompleter.use_jedi = False",
                    "",
                    "from IPython.display import display, Markdown, HTML, Image, Javascript",
                    "from IPython.core.magic import register_line_cell_magic, register_line_magic, register_cell_magic",
                    "display(HTML('<style>.container { width:%d%% !important; }</style>' % 90))",
                    "",
                    "import sys, os, io, time, random, math",
                    "import json, base64, requests, shutil",
                    "import os.path as osp",
                    "import numpy as np",
                    "import argparse, shlex, signal",
                    "",
                    "argparse.ArgumentParser.exit = lambda *arg, **kwargs: _IGNORE_",
                    "",
                    "def _IMPORT(x):",
                    "    try:",
                    "        x = x.strip()",
                    "        if x.startswith('https://'):",
                    "            x = x[8:]",
                    "        if not x.endswith('.py'):",
                    "            x = x + '.py'",
                    "        if x[0] == '/':",
                    "            with open(x) as fr:",
                    "                x = fr.read()",
                    "        else:",
                    "            x = x.replace('blob/main/', '').replace('blob/master/', '')",
                    "            if x.startswith('raw.githubusercontent.com'):",
                    "                uri = 'https://' + x",
                    "                x = requests.get(uri)",
                    "                if x.status_code == 200:",
                    "                    x = x.text",
                    "            elif x.startswith('github.com'):",
                    "                uri = x.replace('github.com', 'raw.githubusercontent.com')",
                    "                mod = uri.split('/')",
                    "                for s in ['main', 'master']:",
                    "                    uri = 'https://' + '/'.join(mod[:3]) + s + '/'.join(mod[-3:])",
                    "                    x = requests.get(uri)",
                    "                    if x.status_code == 200:",
                    "                        x = x.text",
                    "                        break",
                    "            elif x.startswith('gitee.com'):",
                    "                mod = x.split('/')",
                    "                for s in ['/raw/main/', '/raw/master/']:",
                    "                    uri = 'https://' + '/'.join(mod[:3]) + s + '/'.join(mod[3:])",
                    "                    x = requests.get(uri)",
                    "                    if x.status_code == 200:",
                    "                        x = x.text",
                    "                        break",
                    "        exec(x, globals())",
                    "    except:",
                    "        pass",
                    "",
                    "def _DIR(x, dumps=True, ret=True):",
                    "    attrs = sorted([y for y in dir(x) if not y.startswith('_')])",
                    "    result = '%s: %s' % (str(type(x))[8:-2], json.dumps(attrs) if dumps else attrs)",
                    "    if ret:",
                    "        return result",
                    "    print(result)",
                    "",
                ]
            },//}}}
            '---',
            {
                'name': 'Custom Magics',//{{{
                'sub-menu': [
                    {
                        'name': 'Template WriteFile',//{{{
                        'snippet': [
                            "@register_line_cell_magic",
                            "def template_writefile(line, cell):",
                            "    path = os.path.dirname(line)",
                            "    if not os.path.exists(path):",
                            "        os.makedirs(path, exist_ok=True)",
                            "    with open(line, 'w') as fw:",
                            "        fw.write(cell.format(**globals()))",
                            "",
                        ]
                    },//}}}
                    '---',
                    {
                        'name': 'Html Display(*)',//{{{
                        'snippet': [
                            "def display_html(port, height=600):",
                            "    from IPython import display",
                            "    from html import escape as html_escape",
                            "    frame_id = 'erlangai-frame-{:08x}'.format(random.getrandbits(64))",
                            "    shell = '''",
                            "      <iframe id='%HTML_ID%' width='100%' height='%HEIGHT%' frameborder='0'>",
                            "      </iframe>",
                            "      <script>",
                            "        (function() {",
                            "          const frame = document.getElementById(%JSON_ID%);",
                            "          const url = new URL(%URL%, window.location);",
                            "          const port = %PORT%;",
                            "          if (port) {",
                            "            url.port = port;",
                            "          }",
                            "          frame.src = url;",
                            "        })();",
                            "      </script>",
                            "    '''",
                            "    replacements = [",
                            "        ('%HTML_ID%', html_escape(frame_id, quote=True)),",
                            "        ('%JSON_ID%', json.dumps(frame_id)),",
                            "        ('%HEIGHT%', '%d' % height),",
                            "        ('%PORT%', '%d' % port),",
                            "        ('%URL%', json.dumps('/')),",
                            "    ]",
                            "    for (k, v) in replacements:",
                            "        shell = shell.replace(k, v)",
                            "    display.display(display.HTML(shell))",
                            "",
                        ],
                        'sub-menu': [
                            {
                                'name': 'Netron Display',//{{{
                                'snippet': [
                                    "@register_line_magic",
                                    "def netron(line):",
                                    "    parser = argparse.ArgumentParser(prog='netron')",
                                    "    parser.add_argument('--file', '-f', type=str, required=True, help='netron model file')",
                                    "    parser.add_argument('--port', '-p', type=int, default=0, help='netron server port')",
                                    "    parser.add_argument('--height', type=int, default=500, help='display netron html window hight')",
                                    "    import netron",
                                    "    try:",
                                    "        args = parser.parse_args(shlex.split(line))",
                                    "        address = netron.start(args.file, address=('0.0.0.0', args.port), browse=False)",
                                    "        display_html(address[1], args.height)",
                                    "    except:",
                                    "        pass",
                                    "",
                                ],
                            }, //}}}
                            {
                                'name': 'Tensorboard Display',//{{{
                                'snippet': [
                                    "@register_line_magic",
                                    "def tensorboard(line):",
                                    "    parser = argparse.ArgumentParser(prog='tensorboard')",
                                    "    parser.add_argument('--logdir', '-d', type=str, required=True, help='tensorboard logdir')",
                                    "    parser.add_argument('--port', '-p', type=int, required=True, help='tensorboard server port')",
                                    "    parser.add_argument('--height', type=int, default=600, help='display netron html window hight')",
                                    "    from tensorboard import manager as tbmanager",
                                    "",
                                    "    infos = tbmanager.get_all()",
                                    "    for info in infos:",
                                    "        if info.port != port: continue",
                                    "        try:",
                                    "            os.kill(info.pid, signal.SIGKILL)",
                                    "            os.unlink(os.path.join(tbmanager._get_info_dir(), f'pid-{info.pid}.info'))",
                                    "        except OSError as e:",
                                    "            if e.errno != errno.ENOENT: raise",
                                    "        except Exception:",
                                    "            pass",
                                    "        break",
                                    "",
                                    "    try:",
                                    "        args = parser.parse_args(shlex.split(line))",
                                    "        strargs = f'--host 0.0.0.0 --port {args.port} --logdir {args.logdir} --reload_interval 10'",
                                    "        command = shlex.split(strargs, comments=True, posix=True)",
                                    "        tbmanager.start(command)",
                                    "        display_html(args.port, args.height)",
                                    "    except:",
                                    "        pass",
                                    ""
                                ]
                            },//}}}
                        ]
                    },//}}}
                ]
            },//}}}
        ]
    };//}}}

    var utils = {//{{{
        'name': 'Utils',
        'sub-menu-direction': 'left',
        'sub-menu': [
            {
                'name': 'Print Progress Bar',//{{{
                'snippet': [
                    "from tqdm.notebook import tqdm",
                    "def print_progress_bar(x):",
                    "    print('\\r', end='')",
                    "    print('Progress: {}%:'.format(x), '%s%s' % ('▋'*(x//2), '.'*((100-x)//2)), end='')",
                    "    sys.stdout.flush()",
                    ""
                ]
            },//}}}
            {
                'name': 'Random Seed',//{{{
                'snippet': [
                    "def  set_rng_seed(x):",
                    "    try:",
                    "        random.seed(x)",
                    "        np.random.seed(x)",
                    "        torch.manual_seed(x)",
                    "    except: ",
                    "        pass",
                    "",
                    "set_rng_seed(888)",
                    ""
                ]
            },//}}}
            {
                'name': 'Image to Base64',//{{{
                'snippet': [
                    "def img2bytes(x, width=None, height=None):",
                    "    if isinstance(x, bytes):",
                    "        return x",
                    "",
                    "    if isinstance(x, str):",
                    "        if os.path.isfile(x):",
                    "            x = PIL.Image.open(x).convert('RGB')",
                    "        else:",
                    "            import cairosvg",
                    "            with io.BytesIO() as fw:",
                    "                cairosvg.svg2png(bytestring=x, write_to=fw,",
                    "                        output_width=width, output_height=height)",
                    "                return fw.getvalue()",
                    "",
                    "    from matplotlib.figure import Figure",
                    "    if isinstance(x, Figure):",
                    "        with io.BytesIO() as fw:",
                    "            plt.savefig(fw)",
                    "            return fw.getvalue()",
                    "",
                    "    from torch import Tensor",
                    "    from torchvision import transforms",
                    "    from PIL import Image",
                    "    if isinstance(x, Tensor):",
                    "        x = transforms.ToPILImage()(x)",
                    "    elif isinstance(x, np.ndarray):",
                    "        x = Image.fromarray(x.astype('uint8')).convert('RGB')",
                    "",
                    "    if isinstance(x, Image.Image):",
                    "        if width and height:",
                    "            x = x.resize((width, height))",
                    "        with io.BytesIO() as fw:",
                    "            x.save(fw, format='PNG')",
                    "            return fw.getvalue()",
                    "    raise NotImplementedError(type(x))",
                    "",
                    "def img2b64(x):",
                    "    return base64.b64encode(img2bytes(x)).decode()",
                    ""
                ]
            },//}}}
            '---',
            {
                'name': 'Display Function(*)',//{{{
                'snippet': [
                        "",
                        "###",
                        "### Display ###",
                        "###",
                        "",
                        "_IMPORT('import pandas as pd')",
                        "_IMPORT('import cv2')",
                        "_IMPORT('from PIL import Image')",
                        "_IMPORT('import matplotlib.pyplot as plt')",
                        "_IMPORT('import plotly')",
                        "_IMPORT('import plotly.graph_objects as go')",
                        "_IMPORT('import ipywidgets as widgets')",
                        "_IMPORT('from ipywidgets import interact, interactive, fixed, interact_manual')",
                        "",
                        "# plotly.offline.init_notebook_mode(connected=False)",
                        "",
                        "plt.rcParams['figure.figsize'] = (12.0, 8.0)",
                        "",
                ],
                'sub-menu': [
                    {
                        'name': 'Image Read',//{{{
                        'snippet': [
                            "",
                            "def im_read(url, rgb=True, size=None):",
                            "    if url.startswith('http'):",
                            "        response = requests.get(url)",
                            "        if response:",
                            "            imgmat = np.frombuffer(response.content, dtype=np.uint8)",
                            "            img = cv2.imdecode(imgmat, cv2.IMREAD_COLOR)",
                            "        else:",
                            "            return None",
                            "    else:",
                            "        img = cv2.imread(url)",
                            "        ",
                            "    if rgb:",
                            "        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)",
                            "    if size:",
                            "        if isinstance(size, int):",
                            "            size = (size, size)",
                            "        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)",
                            "    return img",
                            "",
                        ]
                    },//}}}
                    {
                        'name': 'Show Table(MD)',//{{{
                        'snippet': [
                            "def show_table(headers, data, width=900):",
                            "    from IPython.display import Markdown",
                            "    ncols = len(headers)",
                            "    width = int(width / ncols)",
                            "    lralign = []",
                            "    caption = []",
                            "    for item in headers:",
                            "        astr = ''",
                            "        if item[0] == ':':",
                            "            astr = ':'",
                            "            item = item[1:]",
                            "        astr += '---'",
                            "        if item[-1] == ':':",
                            "            astr += ':'",
                            "            item = item[:-1]",
                            "        lralign.append(astr)",
                            "        caption.append(item)",
                            "    captionstr = '|'.join(caption) + chr(10)",
                            "    lralignstr = '|'.join(lralign) + chr(10)",
                            "    imgholdstr = '|'.join(['<img width=%d/>' % width] * ncols) + chr(10)",
                            "    table = captionstr + lralignstr + imgholdstr",
                            "    is_dict = isinstance(data[0], dict)",
                            "    for row in data:",
                            "        if is_dict:",
                            "            table += '|'.join([f'{row[c]}' for c in caption]) + chr(10)",
                            "        else:",
                            "            table += '|'.join([f'{col}' for col in row]) + chr(10)",
                            "    return Markdown(table)",
                            "",
                        ]
                    },//}}}
                    {
                        'name': 'Show Video',//{{{
                        'snippet': [
                            "def show_video(vidsrc, width=None, height=None):",
                            "    W, H = '', ''",
                            "    if width:",
                            "        W = 'width=%d' % width",
                            "    if height:",
                            "        H = 'height=%d' % height",
                            "    if vidsrc.startswith('http'):",
                            "        data_url = vidsrc",
                            "    else:",
                            "        mp4 = open(vidsrc, 'rb').read()",
                            "        data_url = 'data:video/mp4;base64,' + base64.b64encode(mp4).decode()",
                            "    return HTML('<center><video %s %s controls src=\"%s\" type=\"video/mp4\"/></center>' % (W, H, data_url))",
                            "",
                        ]
                    },//}}}
                    {
                        'name': 'Show Image',//{{{
                        'snippet': [
                            "def show_image(imgsrc, width=None, height=None):",
                            "    if isinstance(imgsrc, np.ndarray):",
                            "        img = imgsrc",
                            "        if width or height:",
                            "            if width and height:",
                            "                size = (width, height)",
                            "            else:",
                            "                rate = img.shape[1] / img.shape[0]",
                            "                if width:",
                            "                    size = (width, int(width/rate))",
                            "                else:",
                            "                    size = (int(height*rate), height)",
                            "            img = cv2.resize(img, size)",
                            "            plt.figure(figsize=(3*int(size[0]/80+1), 3*int(size[1]/80+1)), dpi=80)",
                            "        plt.axis('off')",
                            "        if len(img.shape) > 2:",
                            "            plt.imshow(img);",
                            "        else:",
                            "            plt.imshow(img, cmap='gray');",
                            "        return",
                            "",
                            "    W, H = '', ''",
                            "    if width:",
                            "        W = 'width=%d' % width",
                            "    if height:",
                            "        H = 'height=%d' % height",
                            "    if imgsrc.startswith('http'):",
                            "        data_url = imgsrc",
                            "    else:",
                            "        if len(imgsrc) > 2048:",
                            "            data_url = 'data:image/jpg;base64,' + imgsrc",
                            "        else:",
                            "            img = open(imgsrc, 'rb').read()",
                            "            data_url = 'data:image/jpg;base64,' + base64.b64encode(img).decode()",
                            "    return HTML('<center><img %s %s src=\"%s\"/></center>' % (W, H, data_url))",
                            "",
                        ]
                    },//}}}
                ],
            },//}}}
        ]
    };//}}}

    var erlangai = {//{{{
        'name' : 'ErlangAI',
        'sub-menu-direction': 'left',
        'sub-menu' : [
            {
                'name': 'Pytorch(*)',
                'snippet': [
                    "",
                    "###",
                    "### Torch ###",
                    "###",
                    "",
                    "import torch",
                    "import torch.nn as nn",
                    "import torch.nn.functional as F",
                    "import torch.optim as O",
                    "from torchvision import models as M",
                    "from torchvision import transforms as T",
                    "from torch.utils.data import Dataset, DataLoader",
                    ""
                ],
                'sub-menu': [
                    {
                        'name': 'Uknow',
                        'snippet': [
                        ],
                    }
                ]
            },
            {
                'name': 'Tensorflow(*)',//{{{
                'snippet': [
                    "",
                    "###",
                    "### Tensorflow ###",
                    "###",
                    "",
                    "import tensorflow as tf",
                    "import tf2onnx",
                    "",
                ]
            },//}}}
            '---',
            {
                'name': 'Onnx(*)',//{{{
                'snippet': [
                    "",
                    "###",
                    "### Onnx ###",
                    "###",
                    "",
                    "import onnx",
                    "import onnx.helper as OH",
                    "import onnxruntime as rt",
                    "",
                ]
            }//}}}
        ],
    };//}}}

    var markdown = {//{{{
        'name' : 'Markdown',
        'sub-menu-direction': 'left',
        'sub-menu' : [
            {
                'name': 'Alert',//{{{
                'sub-menu': [
                    {
                        'name': 'Info',
                        'snippet': [
                            '<div class="alert alert-info">',
                            '',
                            '</div>',
                        ]
                    },
                    {
                        'name': 'Warning',
                        'snippet': [
                            '<div class="alert alert-warning">',
                            '',
                            '</div>',
                        ]
                    },
                    {
                        'name': 'Danger',
                        'snippet': [
                            '<div class="alert alert-danger">',
                            '',
                            '</div>',
                        ]
                    },
                    {
                        'name': 'Success',
                        'snippet': [
                            '<div class="alert alert-success">',
                            '',
                            '</div>',
                        ]
                    },
                ]//}}}
            },
            {
                'name': 'Emoji(*)',//{{{
                'snippet': [
                    '<div style="margin-top:30px; width:100%;">',
                    '    <div style="float:left;font-size:18px;">',
                    '        &#128279;&#128285;&#9757;',
                    '    </div>',
                    '</div>',
                ],
                'sub-menu': [
                    {
                        'name': 'Link&Top',
                        'snippet': [
                            '<div style="margin-top:30px; width: 100%;">',
                            '    <div style="float:left;">',
                            '        <a',
                            '           title=""',
                            '           href=""',
                            '           style="color:blue;font-size:28px;text-decoration:none;">',
                            '                &#128279;',
                            '        </a>',
                            '    </div>',
                            '    <div style="text-align:right">',
                            '        <a',
                            '           title="Back to Top"',
                            '           href="#Contents"',
                            '           onclick="window.scrollTo(0, 0);"',
                            '           style="color:blue;font-size:28px;text-decoration:none;">',
                            '               &#128285;',
                            '        </a>',
                            '    </div>',
                            '</div>',
                        ]
                    },
                ]//}}}
            },
        ]
    };//}}}

    // snippets_menu.options['menus'].push(snippets_menu.default_menus[0]);
    snippets_menu.options['menus'].push(magics);
    snippets_menu.options['menus'].push(utils);
    snippets_menu.options['menus'].push(erlangai);
    snippets_menu.options['menus'].push(markdown);
    // console.log(snippets_menu)
    console.log('Loaded `snippets_menu` customizations from `custom.js`');
});
