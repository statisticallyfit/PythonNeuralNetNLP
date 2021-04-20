import os

c = get_config()
custom_rel_path = os.path.join("../../../../../../../../../bin/python/miniconda3/envs/pymatrix_env/share/jupyter/nbconvert/templates/")
custom_abs_path = os.path.join("/development/bin/python/miniconda3/envs/pymatrix_env/share/jupyter/nbconvert/templates/") 

custom_latex_abs_path = os.path.join(custom_abs_path, "latex/")
custom_html_abs_path = os.path.join(custom_abs_path, "html/")

c.TemplateExporter.template_path.append(custom_abs_path)
c.TemplateExporter.template_path.append(custom_latex_abs_path)
c.TemplateExporter.template_path.append(custom_html_abs_path)
c.TemplateExporter.template_path.append(custom_rel_path)

c.LatexExporter.template_file = 'classic'
