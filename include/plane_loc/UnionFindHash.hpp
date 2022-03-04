//
// Created by janw on 19.10.2021.
//

#ifndef PLANE_LOC_UNIONFINDHASH_HPP
#define PLANE_LOC_UNIONFINDHASH_HPP

#include <vector>
#include <unordered_map>

/** \brief Struktura reprezentująca węzeł w klasie UnionFind.
 *
 */
template<class T>
struct SetNodeHash{
    int parent, rank, nsize;
    T key;
    SetNodeHash(const T &ikey) : parent(-1), rank(0), nsize(1), key(ikey) {}
    SetNodeHash(int iparent, int irank, int insize) : parent(iparent), rank(irank), nsize(insize) {}
};

/** \brief Klasa reprezentująca rozłączne zbiory, umożliwiająca
 * 			efektywne ich łączenie.
 */
template<class T>
class UnionFindHash{
public:
    UnionFindHash();
    ~UnionFindHash();

    /** \brief Funkcja znajdująca id zbioru, do którego należy węzeł node.
     *
     */
    T findSet(const T &key) {
        int node = getNode(key);
        if(set[node].parent == -1){
            return node;
        }

        int root = findRoot(set[node].parent);
        set[node].parent = root;
        return set[root].key;
    }

    /** \brief Funkcja łącząca dwa zbiory.
     *
     */
    T unionSets(const T &key1, const T &key2) {
        int node1 = getNode(key1);
        int node2 = getNode(key2);

        int node1Root = findRoot(node1);
        int node2Root = findRoot(node2);
        if(set[node1Root].rank > set[node2Root].rank){
            set[node2Root].parent = node1Root;
            set[node1Root].nsize += set[node2Root].nsize;
            return set[node1Root].key;
        }
        else if(set[node1Root].rank < set[node2Root].rank){
            set[node1Root].parent = node2Root;
            set[node2Root].nsize += set[node1Root].nsize;
            return set[node2Root].key;
        }
        else if(node1Root != node2Root){
            set[node2Root].parent = node1Root;
            set[node1Root].rank++;
            set[node1Root].nsize += set[node2Root].nsize;
            return set[node1Root].key;
        }
        return set[node1Root].key;
    }

    /** \brief Funkcja zwracająca rozmiar zbioru.
     *
     */
    int size(const T &key) {
        int node = getNode(key);
        return set[findRoot(node)].nsize;
    }
private:
    int getNode(const T &key) {
        if (map.count(key) == 0) {
            set.template emplace_back(key);
            map[key] = set.size() - 1;
        }
        return map[key];
    }

    int findRoot(int node) {
        if(set[node].parent == -1){
            return node;
        }

        set[node].parent = findSet(set[node].parent);
        return set[node].parent;
    }

    std::vector<SetNodeHash<T>> set;
    std::unordered_map<T, int> map;
};

#endif //PLANE_LOC_UNIONFINDHASH_HPP
