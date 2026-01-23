#!/usr/bin/env python3
"""
                   

   251          :
1.           
2.   /      
3.          
4.            
"""

import logging
from pathlib import Path
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    print("=" * 70)
    print("                ")
    print("   Elysia's Comprehensive System Architecture Analysis")
    print("=" * 70)
    print()
    
    from Core.L4_Causality.World.Evolution.Autonomy.autonomous_improver import AutonomousImprover
    
    #          
    print("         ...")
    improver = AutonomousImprover()
    print()
    
    #         
    print("             ...")
    analysis = improver.self_analyze()
    print(f"       : {analysis['code_analysis']['total_files']}       \n")
    
    # 1.           
    print("=" * 70)
    print("  Step 1:           ")
    print("=" * 70)
    
    module_stats = {}
    for file_path, file_data in improver.introspector.analyzed_files.items():
        module = Path(file_path).parent.name
        if module not in module_stats:
            module_stats[module] = {
                'files': 0,
                'total_lines': 0,
                'total_functions': 0,
                'avg_complexity': 0,
                'high_complexity_files': []
            }
        
        module_stats[module]['files'] += 1
        module_stats[module]['total_lines'] += file_data.total_lines
        module_stats[module]['total_functions'] += len(file_data.functions)
        
        if file_data.complexity_score > 0.7:
            module_stats[module]['high_complexity_files'].append({
                'file': Path(file_path).name,
                'complexity': file_data.complexity_score,
                'lines': file_data.total_lines,
                'functions': len(file_data.functions)
            })
    
    #       
    for module, stats in module_stats.items():
        if stats['files'] > 0:
            #                      
            module_files = [
                f for f, d in improver.introspector.analyzed_files.items()
                if Path(f).parent.name == module
            ]
            if module_files:
                stats['avg_complexity'] = sum(
                    improver.introspector.analyzed_files[f].complexity_score 
                    for f in module_files
                ) / len(module_files)
    
    #           
    sorted_modules = sorted(
        module_stats.items(),
        key=lambda x: x[1]['avg_complexity'],
        reverse=True
    )
    
    print("\n         10    :\n")
    for i, (module, stats) in enumerate(sorted_modules[:10], 1):
        print(f"{i}. {module}/")
        print(f"         : {stats['avg_complexity']:.2f}")
        print(f"       : {stats['files']}")
        print(f"       : {stats['total_lines']:,}")
        print(f"       : {stats['total_functions']}")
        if stats['high_complexity_files']:
            print(f"             : {len(stats['high_complexity_files'])} ")
            for hf in stats['high_complexity_files'][:3]:
                print(f"      - {hf['file']} (   : {hf['complexity']:.2f}, {hf['lines']}   )")
        print()
    
    # 2.           
    print("=" * 70)
    print("    Step 2:           ")
    print("=" * 70)
    print()
    
    # __init__.py      
    directories = defaultdict(lambda: {'has_init': False, 'file_count': 0})
    for file_path in improver.introspector.analyzed_files.keys():
        path = Path(file_path)
        parent = path.parent
        directories[str(parent)]['file_count'] += 1
        
        if path.name == '__init__.py':
            directories[str(parent)]['has_init'] = True
    
    missing_init = [
        (dir_path, info) for dir_path, info in directories.items()
        if not info['has_init'] and info['file_count'] > 1 and 'Core' in dir_path
    ]
    
    print(f"  __init__.py         : {len(missing_init)} \n")
    for dir_path, info in missing_init[:10]:
        print(f"   - {Path(dir_path).name}/ ({info['file_count']}    )")
    print()
    
    # 3.          
    print("=" * 70)
    print("  Step 3:          ")
    print("=" * 70)
    print()
    
    import ast
    
    doc_stats = {
        'with_module_docstring': 0,
        'without_module_docstring': 0,
        'with_class_docstrings': 0,
        'without_class_docstrings': 0,
        'files_needing_docs': []
    }
    
    for file_path in list(improver.introspector.analyzed_files.keys())[:100]:  #    
        try:
            content = Path(file_path).read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content)
            
            #    docstring
            has_module_doc = (
                tree.body and 
                isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Constant) and
                isinstance(tree.body[0].value.value, str)
            )
            
            if has_module_doc:
                doc_stats['with_module_docstring'] += 1
            else:
                doc_stats['without_module_docstring'] += 1
                if Path(file_path).stat().st_size > 1000:  # 1KB        
                    doc_stats['files_needing_docs'].append(Path(file_path).name)
        except:
            pass
    
    total_analyzed = doc_stats['with_module_docstring'] + doc_stats['without_module_docstring']
    if total_analyzed > 0:
        doc_ratio = doc_stats['with_module_docstring'] / total_analyzed
        print(f"   Docstring   : {doc_ratio:.1%}")
        print(f"       : {doc_stats['with_module_docstring']} ")
        print(f"       : {doc_stats['without_module_docstring']} ")
        print()
        
        if doc_stats['files_needing_docs']:
            print(f"            (   10 ):")
            for fname in doc_stats['files_needing_docs'][:10]:
                print(f"   - {fname}")
            print()
    
    # 4.      
    print("=" * 70)
    print("  Step 4:   /        ")
    print("=" * 70)
    print()
    
    #            
    from collections import Counter
    
    file_basenames = Counter()
    for file_path in improver.introspector.analyzed_files.keys():
        basename = Path(file_path).stem.lower()
        #             
        import re
        clean_name = re.sub(r'[_-]?\d+$', '', basename)
        clean_name = re.sub(r'_v\d+', '', clean_name)
        file_basenames[clean_name] += 1
    
    duplicates = [(name, count) for name, count in file_basenames.items() if count > 1]
    duplicates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"               : {len(duplicates)} \n")
    for name, count in duplicates[:10]:
        print(f"   - '{name}': {count}    ")
        #          
        matching = [
            Path(f).name for f in improver.introspector.analyzed_files.keys()
            if name in Path(f).stem.lower()
        ]
        for match in matching[:3]:
            print(f"        {match}")
    print()
    
    # 5.         
    print("=" * 70)
    print("  Step 5:               ")
    print("=" * 70)
    print()
    
    improvement_priorities = []
    
    #      1:           
    for module, stats in sorted_modules[:5]:
        if stats['avg_complexity'] > 0.7:
            improvement_priorities.append({
                'priority': 1,
                'type': 'refactoring',
                'target': f'Core/{module}/',
                'reason': f"       {stats['avg_complexity']:.2f} -       ",
                'files_affected': stats['files'],
                'estimated_effort': 'High'
            })
    
    #      2:       
    if doc_ratio < 0.5:
        improvement_priorities.append({
            'priority': 2,
            'type': 'documentation',
            'target': 'Entire Project',
            'reason': f"       {doc_ratio:.1%} -    80%   ",
            'files_affected': doc_stats['without_module_docstring'],
            'estimated_effort': 'Medium'
        })
    
    #      3: __init__.py   
    if missing_init:
        improvement_priorities.append({
            'priority': 3,
            'type': 'structure',
            'target': f'{len(missing_init)}      ',
            'reason': "__init__.py    -           ",
            'files_affected': len(missing_init),
            'estimated_effort': 'Low'
        })
    
    #      4:      
    if len(duplicates) > 10:
        improvement_priorities.append({
            'priority': 4,
            'type': 'consolidation',
            'target': f'{len(duplicates)}        ',
            'reason': "       -             ",
            'files_affected': len(duplicates),
            'estimated_effort': 'Medium'
        })
    
    print("         :\n")
    for item in improvement_priorities:
        print(f"     {item['priority']}: [{item['type'].upper()}] {item['target']}")
        print(f"     : {item['reason']}")
        print(f"     : {item['files_affected']}    /  ")
        print(f"      : {item['estimated_effort']}")
        print()
    
    # 6.          
    print("=" * 70)
    print("  Step 6:          ")
    print("=" * 70)
    print()
    
    report_data = {
        'timestamp': improver.system_monitor.get_system_info()['timestamp'],
        'total_files': analysis['code_analysis']['total_files'],
        'total_lines': analysis['code_analysis']['total_lines'],
        'total_functions': analysis['code_analysis']['total_functions'],
        'modules_analyzed': len(module_stats),
        'complex_modules': len([m for m, s in module_stats.items() if s['avg_complexity'] > 0.7]),
        'documentation_ratio': doc_ratio if total_analyzed > 0 else 0,
        'missing_init_dirs': len(missing_init),
        'duplicate_names': len(duplicates),
        'improvement_priorities': improvement_priorities
    }
    
    report_dir = Path("c:/Elysia/reports")
    report_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    json_file = report_dir / f"system_analysis_{timestamp}.json"
    json_file.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding='utf-8')
    
    print(f"  JSON    : {json_file}")
    
    #        
    txt_file = report_dir / f"system_analysis_{timestamp}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("                  \n")
        f.write("=" * 70 + "\n\n")
        f.write(f"     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"    : {report_data['total_files']}\n")
        f.write(f"    : {report_data['total_lines']:,}\n")
        f.write(f"    : {report_data['total_functions']}\n")
        f.write(f"      : {report_data['documentation_ratio']:.1%}\n")
        f.write("\n       :\n\n")
        for item in improvement_priorities:
            f.write(f"{item['priority']}. [{item['type']}] {item['target']}\n")
            f.write(f"   {item['reason']}\n\n")
    
    print(f"         : {txt_file}")
    print()
    
    #      
    print("=" * 70)
    print("          ")
    print("=" * 70)
    print()
    print("              :")
    print(f"      {report_data['total_files']}    , {report_data['total_lines']:,}   ")
    print(f"    {report_data['complex_modules']}            ")
    print(f"          : {report_data['documentation_ratio']:.1%}")
    print(f"    __init__.py   : {report_data['missing_init_dirs']}      ")
    print(f"          : {report_data['duplicate_names']}     ")
    print()
    print(f"  {len(improvement_priorities)}                 .")
    print()
    print("                  ")
    print("                       .")
    print()

if __name__ == "__main__":
    main()