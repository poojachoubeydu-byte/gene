import requests
import time


def fetch_gene_info(gene_symbol):
    try:
        search_url = (
            'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
            f'?db=gene&term={gene_symbol}[Gene+Name]'
            '+AND+Homo+sapiens[Organism]' \
            '&retmode=json&retmax=1'
        )
        r = requests.get(search_url, timeout=6)
        r.raise_for_status()
        ids = r.json().get('esearchresult', {}).get('idlist', [])
        if not ids:
            return {'error': f'Gene {gene_symbol} not found in NCBI'}

        gene_id = ids[0]
        time.sleep(0.4)

        fetch_url = (
            'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
            f'?db=gene&id={gene_id}&retmode=json'
        )
        r2 = requests.get(fetch_url, timeout=6)
        r2.raise_for_status()
        result = r2.json().get('result', {}).get(gene_id, {})

        ncbi_data = {
            'full_name': result.get('description', ''),
            'summary': result.get('summary', 'No summary available.'),
            'location': result.get('maplocation', 'Unknown'),
            'chromosome': result.get('chromosome', 'Unknown'),
            'aliases': result.get('otheraliases', ''),
            'ncbi_id': gene_id
        }

        uni_url = (
            'https://rest.uniprot.org/uniprotkb/search'
            f'?query=gene:{gene_symbol}+AND+organism_id:9606+AND+reviewed:true'
            '&fields=id,cc_function,ft_domain,cc_subcellular_location,cc_disease'
            '&format=json&size=1'
        )
        r3 = requests.get(uni_url, timeout=6)
        r3.raise_for_status()
        results = r3.json().get('results', [])
        if not results:
            uniprot_data = {
                'function': 'No UniProt annotation available.',
                'domains': [],
                'subcellular_location': 'Unknown',
                'disease_associations': [],
                'uniprot_id': ''
            }
        else:
            entry = results[0]
            uniprot_id = entry.get('primaryAccession', '')
            function_text = ''
            domains = []
            subcellular = ''
            diseases = []
            for comment in entry.get('comments', []):
                ctype = comment.get('commentType', '')
                if ctype == 'FUNCTION':
                    texts = comment.get('texts', [])
                    if texts:
                        function_text = texts[0].get('value', '')
                elif ctype == 'SUBCELLULAR LOCATION':
                    locs = comment.get('subcellularLocations', [])
                    if locs:
                        subcellular = ', '.join([
                            loc.get('location', {}).get('value', '')
                            for loc in locs[:3]
                        ])
                elif ctype == 'DISEASE':
                    disease = comment.get('disease', {})
                    name = disease.get('diseaseName', '')
                    if name:
                        diseases.append(name)

            for feature in entry.get('features', []):
                if feature.get('type') == 'Domain':
                    desc = feature.get('description', '')
                    if desc:
                        domains.append(desc)

            uniprot_data = {
                'function': function_text or 'No UniProt annotation available.',
                'domains': domains,
                'subcellular_location': subcellular,
                'disease_associations': diseases,
                'uniprot_id': uniprot_id
            }

        return {
            'full_name': ncbi_data['full_name'],
            'summary': ncbi_data['summary'],
            'location': ncbi_data['location'],
            'aliases': ncbi_data['aliases'],
            'function': uniprot_data['function'],
            'domains': uniprot_data['domains'],
            'subcellular_location': uniprot_data['subcellular_location'],
            'disease_associations': uniprot_data['disease_associations'],
            'ncbi_id': ncbi_data['ncbi_id'],
            'uniprot_id': uniprot_data['uniprot_id']
        }
    except Exception as e:
        return {'error': str(e)[:150]}
