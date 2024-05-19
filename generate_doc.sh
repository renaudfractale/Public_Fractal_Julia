pandoc -t markdown_strict --extract-media='./' ./00_source_doc/main.docx -o READM_temp.md
pandoc -t markdown_strict --extract-media='./'  READM.md  -o ./00_source_doc/main_temp.docx