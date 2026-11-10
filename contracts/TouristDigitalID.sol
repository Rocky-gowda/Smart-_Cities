// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title TouristDigitalID
/// @notice Blockchain-based digital tourist ID and incident log for the Smart Tourist
///         Safety Monitoring & Incident Response System. Solidity version of the
///         Python implementation in src/blockchain.py, for deployment on an
///         Ethereum-compatible permissioned network (e.g. Hyperledger Besu, Polygon).
/// @dev    Only hashes are stored on-chain; personal data stays in the issuer's off-chain vault.
contract TouristDigitalID {
    enum Risk { LOW, MODERATE, HIGH, CRITICAL }

    struct TouristRecord {
        bytes32 credentialHash;   // SHA-256 of the signed credential the tourist carries
        bytes32 kycHash;          // salted hash of KYC data (name, document, phone)
        bytes32 itineraryHash;    // hash of the declared itinerary
        uint64 validFrom;         // unix seconds
        uint64 validTo;           // unix seconds
        address issuer;           // authority that registered the ID
        bool revoked;
        bool exists;
    }

    struct Incident {
        bytes32 touristId;
        uint64 timestamp;
        int32 latE4;              // latitude  x 10^4 (about 11 m precision)
        int32 lonE4;              // longitude x 10^4
        Risk risk;
        bool decidedByCloud;
        bytes32 alertHash;        // SHA-256 of the full alert record
        address reporter;
    }

    address public admin;
    mapping(address => bool) public isAuthority;
    mapping(bytes32 => TouristRecord) private records;
    Incident[] private incidents;
    mapping(bytes32 => uint256[]) private incidentsOf;

    event AuthorityAdded(address indexed authority);
    event AuthorityRemoved(address indexed authority);
    event TouristRegistered(bytes32 indexed touristId, address indexed issuer, uint64 validFrom, uint64 validTo);
    event TouristRevoked(bytes32 indexed touristId, address indexed by, string reason);
    event IncidentRecorded(uint256 indexed incidentId, bytes32 indexed touristId, Risk risk, bytes32 alertHash);

    modifier onlyAdmin() {
        require(msg.sender == admin, "not admin");
        _;
    }

    modifier onlyAuthority() {
        require(isAuthority[msg.sender], "not an authority");
        _;
    }

    constructor() {
        admin = msg.sender;
        isAuthority[msg.sender] = true;
        emit AuthorityAdded(msg.sender);
    }

    // ------------------------------------------------------------------ authorities
    function addAuthority(address a) external onlyAdmin {
        require(!isAuthority[a], "already an authority");
        isAuthority[a] = true;
        emit AuthorityAdded(a);
    }

    function removeAuthority(address a) external onlyAdmin {
        require(isAuthority[a], "not an authority");
        isAuthority[a] = false;
        emit AuthorityRemoved(a);
    }

    // ------------------------------------------------------------------ digital ID
    function registerTourist(
        bytes32 touristId,
        bytes32 credentialHash,
        bytes32 kycHash,
        bytes32 itineraryHash,
        uint64 validFrom,
        uint64 validTo
    ) external onlyAuthority {
        require(validTo >= validFrom, "invalid validity period");
        TouristRecord storage r = records[touristId];
        require(!r.exists || r.revoked, "ID already registered");
        records[touristId] = TouristRecord(credentialHash, kycHash, itineraryHash, validFrom, validTo,
                                           msg.sender, false, true);
        emit TouristRegistered(touristId, msg.sender, validFrom, validTo);
    }

    function revokeTourist(bytes32 touristId, string calldata reason) external onlyAuthority {
        TouristRecord storage r = records[touristId];
        require(r.exists, "unknown ID");
        require(!r.revoked, "already revoked");
        r.revoked = true;
        emit TouristRevoked(touristId, msg.sender, reason);
    }

    /// @notice Called by a police / verifier app after scanning the tourist's QR code.
    /// @return valid  true if the presented credential matches, is not revoked and is in date
    /// @return reason a short explanation
    function verifyTourist(bytes32 touristId, bytes32 presentedCredentialHash)
        external
        view
        returns (bool valid, string memory reason)
    {
        TouristRecord storage r = records[touristId];
        if (!r.exists) return (false, "ID not found");
        if (r.credentialHash != presentedCredentialHash) return (false, "credential does not match record");
        if (r.revoked) return (false, "ID revoked");
        if (block.timestamp < r.validFrom || block.timestamp > r.validTo) return (false, "ID not valid today");
        return (true, "valid");
    }

    function getTourist(bytes32 touristId)
        external
        view
        returns (uint64 validFrom, uint64 validTo, address issuer, bool revoked, bool exists)
    {
        TouristRecord storage r = records[touristId];
        return (r.validFrom, r.validTo, r.issuer, r.revoked, r.exists);
    }

    // ------------------------------------------------------------------ incidents
    function recordIncident(
        bytes32 touristId,
        int32 latE4,
        int32 lonE4,
        Risk risk,
        bool decidedByCloud,
        bytes32 alertHash
    ) external onlyAuthority returns (uint256 incidentId) {
        require(records[touristId].exists, "unknown ID");
        incidentId = incidents.length;
        incidents.push(Incident(touristId, uint64(block.timestamp), latE4, lonE4, risk, decidedByCloud,
                                alertHash, msg.sender));
        incidentsOf[touristId].push(incidentId);
        emit IncidentRecorded(incidentId, touristId, risk, alertHash);
    }

    function incidentCount() external view returns (uint256) {
        return incidents.length;
    }

    function getIncident(uint256 incidentId) external view returns (Incident memory) {
        require(incidentId < incidents.length, "no such incident");
        return incidents[incidentId];
    }

    function incidentsOfTourist(bytes32 touristId) external view returns (uint256[] memory) {
        return incidentsOf[touristId];
    }
}
