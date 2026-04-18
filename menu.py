"""
menu.py
Nuke toolbar registration and node lifecycle callbacks for DBFluxFill.
Loaded automatically by Nuke when DBFluxFill is on the plugin path.
"""

import nuke
import os

_menu_dir = os.path.dirname(os.path.abspath(__file__))
nuke.pluginAddPath(os.path.join(_menu_dir, "icons"))

# ---------------------------------------------------------------------------
# Toolbar registration
# ---------------------------------------------------------------------------

toolbar = nuke.menu("Nodes")
db_menu = toolbar.addMenu("DBFluxFill", "DBFluxFill.png")
db_menu.addCommand("DBFluxFill", lambda: _create_dbfluxfill_node())
# db_menu.addCommand("DBFluxFill", lambda: _create_dbfluxfill_node(), icon="")


# ---------------------------------------------------------------------------
# Node creation
# ---------------------------------------------------------------------------

def _create_dbfluxfill_node():
    """
    Load the DBFluxFill group from the .nk file and place it in the graph.
    """
    import DBFluxFill.callbacks as cb
 
    gizmo_dir = cb._get_gizmo_dir()
    nk_path   = os.path.join(gizmo_dir, "DBFluxFill.nk")
 
    if not os.path.isfile(nk_path):
        nuke.message(
            "DBFluxFill: DBFluxFill.nk not found.\n\n"
            "Expected:\n{}".format(nk_path)
        )
        return
 
    nuke.nodePaste(nk_path)
