"""Explicit manufacturing-candidate routes, separate from simulation runs."""
import json
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse
from organic_motor.construct.casting_material import material_template, validate_material


def _read_hash(output):
    """Read design_hash from manifest, or empty string if unavailable."""
    try:
        m=json.loads((Path(output)/'manifest.json').read_text())
        return m.get('design_hash','')
    except Exception:
        return ''


def install(app, root):
    output=Path(root)/'wound_prototype'

    @app.get('/prototype')
    def page():
        return FileResponse(Path(__file__).parent/'static'/'prototype.html',headers={'Cache-Control':'no-store'})

    @app.get('/api/prototype/material-template')
    def template():return material_template()

    @app.post('/api/prototype/material-check')
    def check(card: dict):
        try:return validate_material(card)
        except (ValueError,TypeError,AttributeError):raise HTTPException(422,'材料卡字段格式不正确')

    @app.get('/api/prototype/{filename}')
    def artifact(filename: str):
        allowed={'assembly.glb','molds.glb','manifest.json','manufacturing-kit.zip',
                 'winding_route_mm.csv','assembly-audit.json','shape-screen.json','wiring.json'}
        if filename not in allowed:
            raise HTTPException(404,'未知制造文件')
        path=output/filename
        if not path.is_file():
            raise HTTPException(404,'先运行 python -m organic_motor.construct.wound_prototype --out '+str(output))
        return FileResponse(path,headers={'Cache-Control':'no-cache',
            'X-Design-Hash':_read_hash(output)})
